"""Извлечение графа классификацией пунктов (режим `extract_llm --mode classify`).

Свободное извлечение (модель сама пишет узлы, id и связи) дало F1 0,37: модель
придумывает id, копирует узлы справочника, зацикливается и обрывается.
Здесь модель только выбирает из перечислений, а граф строит код:

- единица работы — один пункт договора (indexing.chunk.clauses);
- модель видит текст пункта, список пунктов этого документа и справочник терминов
  и событий для типа продукта; эталон и YAML она не видит;
- ответ — до 3 элементов {kind, term, event, event_relation, parent_clause}, все поля —
  перечисления в JSON-схеме Ollama, поэтому лишнего или выдуманного id быть не может;
- id назначает код: `<doc_id>__<пункт>` (несколько узлов в пункте — `__<пункт>_<k>`,
  элементы перечня исключений — всегда `__<пункт>_<k>`); название — из термина
  справочника или первые слова пункта, значение — текст пункта;
- связи проверяются по indexing.schema, недопустимые отбрасываются и считаются.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from indexing.chunk import clauses, parse_doc
from indexing.schema import check_edge
from rag import llm
from rag.text import contains_all, tokens

KINDS = ["condition", "fee", "exception", "none"]
RELATIONS = ["cancels", "triggers", "none"]
MAX_ITEMS = 3
MAX_TOKENS = 220  # 3 элемента по ~60 токенов; на CPU ~5 ток/с это до ~45 с на пункт
LABEL = {"condition": "Condition", "fee": "Fee", "exception": "Exception"}

PROMPT = """Ты размечаешь пункт банковского договора для графа знаний. Ничего не пиши, кроме JSON по схеме.

Продукт: {product}. Пункты этого документа:
{clause_list}

Термины (term): {terms}
События (event): {events}

Для пункта верни список items (1–3 элемента), по одному на каждое условие, платёж или связь с событием:
- kind: condition — условие продукта (лимит, ставка, льготный период, кэшбэк, срок, минимальный платёж);
  fee — платёж клиента (обслуживание, комиссия, штраф, неустойка, подписка);
  exception — на что условие НЕ распространяется (перечень исключений);
  none — пункт только описывает событие и его последствие для другого пункта.
- term — термин, которым является это условие или платёж, или null.
- event и event_relation — если в пункте сказано, что событие отменяет условие (cancels)
  или влечёт штраф, ставку или другое последствие (triggers); иначе null и none.
- parent_clause — номер пункта, к которому относится элемент: для exception — пункт с условием,
  из которого исключение; для cancels — пункт с отменяемым условием; иначе null.

Пункт {clause_id} (раздел «{section}»):
{text}"""

PROMPT_HASH = hashlib.sha256(PROMPT.encode()).hexdigest()[:8]

# Проверенный и отвергнутый вариант (промпт v2, хеш 52d8dfb4): подсказки-ключевые слова
# («обычно один элемент», «event — только если назван в пункте», «не начисляется на — exception»).
# На выборке без правил кода он лучше (строгий F1 0,61 против 0,54), но вместе с правилами
# хуже (0,63 против 0,66): модель сама отмечает исключения, но с неверным родительским
# пунктом. См. review/eval_extraction_classify_v*.txt. Оставлен промпт v1.
REJECTED_HINTS = """Подсказки:
- обычно в пункте ОДИН элемент; второй и третий — только если в пункте несколько разных платежей
  или условий («штраф … и неустойка …») либо отдельное последствие события;
- event указывай, только если событие прямо названо в тексте этого пункта, иначе null;
- «не начисляется на», «не распространяется на», «не входят» — это exception;
- «отменяется», «прекращается», «больше нет» — cancels; «штраф», «неустойка», «начисляется … годовых»,
  «переводится на» после события — triggers.

"""


def schema(term_ids: list[str], event_ids: list[str], clause_ids: list[str]) -> dict:
    item = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": KINDS},
            "term": {"type": ["string", "null"], "enum": term_ids + [None]},
            "event": {"type": ["string", "null"], "enum": event_ids + [None]},
            "event_relation": {"type": "string", "enum": RELATIONS},
            "parent_clause": {"type": ["string", "null"], "enum": clause_ids + [None]},
        },
        "required": ["kind", "term", "event", "event_relation", "parent_clause"],
    }
    return {"type": "object", "properties": {"items": {"type": "array", "items": item, "minItems": 1,
                                                       "maxItems": MAX_ITEMS}}, "required": ["items"]}


@dataclass
class Stats:
    clauses: int = 0
    failed: list[str] = field(default_factory=list)  # битый JSON / Ollama недоступен
    invalid_values: int = 0  # значение вне перечисления (если модель или кэш его всё же выдали)
    dropped_edges: list[str] = field(default_factory=list)  # отброшены проверкой схемы или без адресата

    def as_dict(self) -> dict:
        return {"clauses": self.clauses, "failed": self.failed, "invalid_values": self.invalid_values,
                "dropped_edges": self.dropped_edges}


def clean_item(item: dict, term_ids: set, event_ids: set, clause_ids: set, stats: Stats) -> dict:
    """Значение вне перечисления не роняет разбор: заменяется на null/none и считается."""
    def pick(key, allowed, default):
        v = item.get(key, default)
        if v is not None and v not in allowed:
            stats.invalid_values += 1
            return default
        return v
    return {"kind": pick("kind", set(KINDS), "none"), "term": pick("term", term_ids, None),
            "event": pick("event", event_ids, None), "event_relation": pick("event_relation", set(RELATIONS), "none"),
            "parent_clause": pick("parent_clause", clause_ids, None)}


def exception_elements(text: str, events: dict[str, list[str]]) -> list[tuple[str, str | None]]:
    """Элементы перечня исключений и событие каждого (по алиасам справочника, это делает код).

    «…не распространяется на: снятие наличных; переводы с карты.» — элементы через «;».
    Без двоеточия — по событию на каждое найденное («Снятие наличных и переводы … не входят»),
    иначе весь пункт — один элемент.
    """
    def event_of(fragment: str) -> list[str]:
        words = tokens(fragment)
        return [e for e, phrases in events.items() if any(contains_all(p, words) for p in phrases)]

    if ":" in text:
        parts = [p.strip(" .") for p in text.split(":", 1)[1].split(";") if p.strip(" .")]
        return [(p, (event_of(p) or [None])[0]) for p in parts]
    found = event_of(text)
    if len(found) > 1:
        return [(events[e][0], e) for e in found]
    return [(text.strip(" ."), found[0] if found else None)]


# «не начисляется на: …», «не распространяется на …», «… не входят» / «не входят в …».
# Просто «не начисляются» (п. 3.1 «100 дней»: «проценты … не начисляются») — не исключение.
EXCEPTION_CUE = re.compile(r"не\s+(?:распространя\w*|начисля\w*)\s+на\b|не\s+вход\w*(?:\s+в\b|\s*\.?\s*$)",
                           re.IGNORECASE)


def refine(clause_list: list[dict], answers: dict[str, list[dict]], events: dict[str, list[str]]) -> dict:
    """Правила кода поверх ответов модели (слой отделён, чтобы мерить его вклад: --no-rules).

    На выборке модель почти всегда заполняла все 3 элемента: придумывала события, которых
    в пункте нет, дублировала условия и не узнавала перечни исключений. Правила:
    1. событие засчитывается, только если оно названо в тексте пункта (алиасы справочника);
    2. одинаковые (kind, term) внутри пункта схлопываются;
    3. «не начисляется / не распространяется / не входят» — пункт-исключение: элементы
       модели заменяются одним exception; родитель — ближайший предыдущий пункт с тем же
       термином, что модель дала этим элементам, иначе ближайший предыдущий пункт-условие.
    """
    text = {c["id"]: c["text"] for c in clause_list}
    order = [c["id"] for c in clause_list]
    out: dict[str, list[dict]] = {}
    for cid in order:
        items = answers.get(cid)
        if items is None:
            continue
        words = tokens(text[cid])
        named = {e for e, phrases in events.items() if any(contains_all(p, words) for p in phrases)}
        clean, seen = [], set()
        for it in items:
            it = dict(it)
            if it["event"] and it["event"] not in named:
                it["event"], it["event_relation"] = None, "none"
            key = (it["kind"], it["term"], it["event"], it["event_relation"])
            if key in seen or (it["kind"] == "none" and not it["event"]):
                continue
            seen.add(key)
            clean.append(it)
        if EXCEPTION_CUE.search(text[cid]) and not any(it["kind"] == "exception" for it in clean):
            terms = {it["term"] for it in items if it["term"]}
            earlier = order[:order.index(cid)][::-1]
            parent = next((c for c in earlier if any(it["term"] in terms and it["kind"] == "condition"
                                                     for it in out.get(c, []))), None)
            parent = parent or next((c for c in earlier if any(it["kind"] == "condition" for it in out.get(c, []))), None)
            clean = [{"kind": "exception", "term": None, "event": None, "event_relation": "none", "parent_clause": parent}]
        out[cid] = clean
    return out


def build_graph(pid: str, clause_list: list[dict], answers: dict[str, list[dict]], catalog: dict,
                events: dict[str, list[str]], stats: Stats) -> dict:
    """Узлы и связи из ответов классификатора; формат — как у эталона (data/extracted/gold)."""
    nodes, edges = [], []
    by_clause: dict[str, list[dict]] = {}  # пункт -> его узлы (не исключения)
    text = {c["id"]: c["text"] for c in clause_list}

    for c in clause_list:
        items = answers.get(c["id"], [])
        own = [it for it in items if it["kind"] in ("condition", "fee")]
        for k, it in enumerate(own, 1):
            nid = f"{pid}__{c['id']}" if len(own) == 1 else f"{pid}__{c['id']}_{k}"
            name = catalog[it["term"]]["name"] if it["term"] else " ".join(c["text"].split()[:4])
            node = {"id": nid, "label": LABEL[it["kind"]], "name": name, "value": c["text"], "aliases": "",
                    "section": c["id"]}
            nodes.append(node)
            by_clause.setdefault(c["id"], []).append(node)

    def parent_node(clause_id: str | None) -> dict | None:
        cands = by_clause.get(clause_id or "", [])
        return next((n for n in cands if n["label"] == "Condition"), cands[0] if cands else None)

    def add(rel, a, la, b, lb, section):
        err = check_edge(rel, la, lb)
        if err:
            stats.dropped_edges.append(f"{section}: {rel} {a} -> {b}: {err}")
        elif [rel, a, b, section] not in edges:
            edges.append([rel, a, b, section])

    for c in clause_list:
        items = answers.get(c["id"], [])
        own = by_clause.get(c["id"], [])
        for node, it in zip(own, [it for it in items if it["kind"] in ("condition", "fee")]):
            rel = "HAS_CONDITION" if node["label"] == "Condition" else "CHARGES"
            add(rel, pid, "Product", node["id"], node["label"], c["id"])
            if it["term"]:
                add("DEFINED_AS", node["id"], node["label"], it["term"], "Term", c["id"])

        exc_items = [it for it in items if it["kind"] == "exception"]
        if exc_items:
            parent = parent_node(next((it["parent_clause"] for it in exc_items if it["parent_clause"]), None))
            for k, (element, ev) in enumerate(exception_elements(text[c["id"]], events), 1):
                xid = f"{pid}__{c['id']}_{k}"
                nodes.append({"id": xid, "label": "Exception", "name": element, "value": "", "aliases": "",
                              "section": c["id"]})
                if parent is None:
                    stats.dropped_edges.append(f"{c['id']}: HAS_EXCEPTION без пункта-условия")
                else:
                    add("HAS_EXCEPTION", parent["id"], parent["label"], xid, "Exception", c["id"])
                ev = ev or (exc_items[0]["event"] if len(exc_items) == 1 else None)
                if ev:
                    add("APPLIES_TO", xid, "Exception", ev, "Event", c["id"])

        for it in items:
            if not it["event"] or it["event_relation"] == "none":
                continue
            self_node = own[0] if own else None
            if it["event_relation"] == "cancels":
                target = parent_node(it["parent_clause"]) if it["parent_clause"] else (
                    next((n for n in own if n["label"] == "Condition"), None))
                rel = "CANCELS"
            else:
                target = self_node if it["kind"] in ("condition", "fee") and self_node else parent_node(it["parent_clause"])
                rel = "TRIGGERS"
            if target is None:
                stats.dropped_edges.append(f"{c['id']}: {rel} {it['event']} без адресата")
                continue
            add(rel, it["event"], "Event", target["id"], target["label"], c["id"])
    return {"nodes": nodes, "edges": edges}


class ClauseClassifier:
    def __init__(self, catalog: dict, term_ids: list[str], event_ids: list[str], model: str, raw_dir: Path,
                 use_cache: bool = False, rules: bool = True):
        self.catalog, self.term_ids, self.event_ids = catalog, term_ids, event_ids
        self.model, self.raw_dir, self.use_cache, self.rules = model, raw_dir, use_cache, rules

    def classify_doc(self, path: Path) -> tuple[dict, Stats]:
        meta, _ = parse_doc(path)
        pid = meta["doc_id"]
        clause_list = clauses(path)
        ids = [c["id"] for c in clause_list]
        sch = schema(self.term_ids, self.event_ids, ids)
        # Общая часть промпта одинакова для всех пунктов документа и стоит в начале:
        # Ollama переиспользует её KV-кэш, на CPU это заметно быстрее.
        listing = "\n".join(f"{c['id']}. {' '.join(c['text'].split()[:8])}…" for c in clause_list)
        terms = ", ".join(f"{t} ({self.catalog[t]['name']})" for t in self.term_ids)
        evs = ", ".join(f"{e} ({self.catalog[e]['name']})" for e in self.event_ids)
        stats = Stats(clauses=len(clause_list))
        answers = {}
        for c in clause_list:
            prompt = PROMPT.format(product=meta["product"], clause_list=listing, terms=terms, events=evs,
                                   clause_id=c["id"], section=c["section_title"], text=c["text"])
            # в имени — хеш шаблона промпта: ответы на разные версии промпта не смешиваются
            cache = self.raw_dir / f"{pid}_{c['id']}.{re.sub(r'[^a-z0-9]+', '_', self.model.lower())}.{PROMPT_HASH}.json"
            try:
                if self.use_cache and cache.exists():
                    raw = cache.read_text(encoding="utf-8")
                else:
                    raw = llm.chat([{"role": "user", "content": prompt}], model=self.model, schema=sch,
                                   max_tokens=MAX_TOKENS)
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_text(raw, encoding="utf-8")
                items = json.loads(raw)["items"]
                if not isinstance(items, list):
                    raise TypeError("items не список")
            except (llm.LLMUnavailable, json.JSONDecodeError, KeyError, TypeError) as e:
                stats.failed.append(c["id"])
                print(f"  {pid} {c['id']}: ПРОПУЩЕН ({e.__class__.__name__}: {str(e)[:80]})", flush=True)
                continue
            answers[c["id"]] = [clean_item(it, set(self.term_ids), set(self.event_ids), set(ids), stats)
                                for it in items[:MAX_ITEMS] if isinstance(it, dict)]
            print(f"  {pid} {c['id']}: " + "; ".join(
                f"{it['kind']}/{it['term'] or '-'}/{it['event'] or '-'}:{it['event_relation']}"
                f"{'→' + it['parent_clause'] if it['parent_clause'] else ''}" for it in answers[c['id']]), flush=True)
        events = {e: [p for p in self.catalog[e]["aliases"].split("|") if p] for e in self.event_ids}
        final = refine(clause_list, answers, events) if self.rules else answers
        graph = build_graph(pid, clause_list, final, self.catalog, events, stats)
        return {**graph, "stats": stats.as_dict(), "answers": answers, "refined": final, "rules": self.rules}, stats
