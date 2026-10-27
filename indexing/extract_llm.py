"""LLM-экстракция узлов и связей по жёсткой схеме (Ollama + Qwen3).

    python -m indexing.extract_llm                     # все документы
    python -m indexing.extract_llm --only sever_credit # один документ
    python -m indexing.extract_llm --force             # заново, даже если файл продукта уже есть
    python -m indexing.extract_llm --mode classify --only balt_credit sever_install   # классификация пунктов
    python -m eval.eval_extraction --pred data/extracted/llm_classify
    python -m indexing.build_graph --source llm        # затем загрузить в граф
    python -m eval.eval_extraction                     # и сравнить с эталоном

Глоссарий (термины и события) пишется руками и копируется из эталона: это
справочник, к которому LLM привязывает условия, а не предмет экстракции.
"""

import argparse
import json
import re
import shutil
import time
from pathlib import Path

from indexing.chunk import all_docs, chunk_doc, parse_doc
from indexing.classify_clauses import ClauseClassifier
from indexing.schema import EXTRACTED_LABELS, REL_TYPES
from rag import llm

ROOT = Path(__file__).resolve().parent.parent
GOLD_DIR = ROOT / "data" / "extracted" / "gold"
OUT_DIR = ROOT / "data" / "extracted" / "llm"
CLASSIFY_OUT_DIR = ROOT / "data" / "extracted" / "llm_classify"
# Сырые ответы модели по разделам: постобработку можно менять и пересчитывать
# (--reuse-raw) без новых часов генерации на CPU.
RAW_DIR_LLM = ROOT / "data" / "extracted" / "llm_raw"
# Qwen3 8B на CPU (~5 ток/с) при structured output может зациклиться и копировать
# узлы справочника без конца. Поэтому длина ответа и размер массивов ограничены:
# раздел договора — это 1–6 пунктов, 12 узлов и 30 связей с запасом.
MAX_TOKENS = 1200
MAX_NODES, MAX_EDGES = 12, 30

SCHEMA = {
    "type": "object",
    "properties": {
        "nodes": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string"},
            "label": {"type": "string", "enum": EXTRACTED_LABELS},
            "name": {"type": "string"},
            "value": {"type": "string"},
            "section": {"type": "string"},
        }, "required": ["id", "label", "name", "value", "section"]}, "maxItems": MAX_NODES},
        "edges": {"type": "array", "items": {"type": "object", "properties": {
            "type": {"type": "string", "enum": list(REL_TYPES)},
            "from": {"type": "string"},
            "to": {"type": "string"},
            "section": {"type": "string"},
        }, "required": ["type", "from", "to", "section"]}, "maxItems": MAX_EDGES},
    },
    "required": ["nodes", "edges"],
}

PROMPT = """Извлеки из фрагмента банковского документа граф знаний строго по схеме.

Создавай только узлы:
- Condition — условие продукта (лимит, ставка, льготный период, кешбэк, минимальный платёж, срок);
- Fee — платёж клиента (обслуживание, комиссия, штраф, неустойка);
- Exception — исключение из условия (на что условие не распространяется), один узел на каждый пункт перечня.
name — короткое название, value — суть с числами как в тексте, section — номер пункта (например "2.2").

Допустимые связи (from -> to):
{rels}

Уже существующие узлы: ссылайся на них по id в связях, но НЕ создавай для них узлы
(термины и события — не Condition и не Exception). Каждый узел создавай один раз:
- продукт: {product_id} ({product})
- термины (Term): {terms}
- события (Event): {events}
- узлы из предыдущих разделов этого документа: {previous}

Каждое Condition и Fee свяжи с подходящим термином через DEFINED_AS, если такой есть.
Если событие из списка отменяет условие — CANCELS, если включает штраф или новую ставку — TRIGGERS.
Каждое Exception свяжи с условием через HAS_EXCEPTION, а с событием, о котором оно, — через APPLIES_TO
(«снятие сверх лимита» — это событие превышения лимита, а не снятия наличных).
Ничего не выдумывай: только то, что прямо написано во фрагменте.

Фрагмент:
{text}"""


# Чтобы промпт не раздувался, в нём только термины и события, уместные для типа продукта.
DOC_TYPE_TERMS = {
    "credit_card": ["term_grace", "term_psk", "term_minpay", "term_cash_withdrawal", "term_cc_transfer", "term_penalty",
                    "term_fine", "term_credit_limit", "term_grace_extension", "term_subscription", "term_rate",
                    "term_service_fee", "term_cashback", "term_btn_installment", "term_promo"],
    "installment": ["term_installment", "term_bnpl", "term_btn_installment", "term_penalty", "term_fine", "term_rate",
                    "term_early", "term_credit_history", "term_mfo", "term_overdue", "term_service_fee",
                    "term_credit_limit", "term_grace", "term_minpay", "term_cc_transfer", "term_subscription"],
    "debit_card": ["term_service_fee", "term_cash_withdrawal", "term_cashback", "term_cashback_limit",
                   "term_subscription", "term_balance_interest", "term_sbp", "term_loyalty", "term_promo"],
}
DOC_TYPE_TERMS["student_card"] = DOC_TYPE_TERMS["debit_card"] + ["term_student_card"]
DOC_TYPE_EVENTS = {
    "credit_card": ["ev_missed_payment", "ev_not_full_repay", "ev_cash", "ev_transfer", "ev_paid_service",
                    "ev_promo_end", "ev_limit_exceeded"],
    "installment": ["ev_missed_payment", "ev_missed_two", "ev_paid_service"],
    "debit_card": ["ev_low_spend", "ev_paid_service", "ev_promo_end", "ev_limit_exceeded", "ev_cash", "ev_transfer"],
    "student_card": ["ev_student_lost", "ev_low_spend", "ev_cash", "ev_transfer"],
}
SKIP_TYPES = {"glossary", "regulatory", "case"}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9а-я]+", "_", text.lower()).strip("_")


def _canonical(ref: str, pid: str, catalog: dict) -> str | None:
    """Ссылку модели на продукт, термин или событие приводит к id справочника.

    Qwen3 часто создаёт их заново: «product_balt_credit», «event_ev_limit_exceeded»,
    «balt_credit__term_grace» — это тот же продукт и те же узлы справочника.
    """
    r = ref[len(pid) + 2:] if ref.startswith(f"{pid}__") else ref
    if r in catalog or r == pid:
        return r
    if r in {"product", f"product_{pid}", f"{pid}_product"}:
        return pid
    return next((cid for cid in catalog if r.endswith(cid)), None)


def _merge(data: dict, pid: str, catalog: dict, nodes: list, edges: list) -> tuple[int, int]:
    """Добавляет узлы и связи раздела в документ; возвращает (новых узлов, новых связей)."""
    remap = {}
    seen = {(n["label"], n["name"], n["value"], n["section"]) for n in nodes}
    names = {catalog[i]["name"].lower() for i in catalog}
    added = 0
    for n in data["nodes"]:
        canon = _canonical(n["id"], pid, catalog)
        if canon or n["name"].lower() == "продукт":
            remap[n["id"]] = canon or pid  # копия продукта/термина/события — только ссылка
            continue
        if n["name"] in REL_TYPES:  # «узел» DEFINED_AS — модель перепутала узел со связью
            continue
        if n["label"] != "Exception" and n["name"].lower() in names and not n["value"]:
            continue
        new_id = f"{pid}__{_slug(n['id'])}"
        remap[n["id"]] = new_id
        key = (n["label"], n["name"], n["value"], n.get("section", ""))
        if key in seen:  # повтор: на нём модель и зацикливается
            continue
        seen.add(key)
        nodes.append({**n, "id": new_id, "aliases": ""})
        added += 1
    before = len(edges)
    for e in data["edges"]:
        ends = [remap.get(x) or _canonical(x, pid, catalog) or x for x in (e["from"], e["to"])]
        edge = [e["type"], *ends, e.get("section", "")]
        if edge not in edges:
            edges.append(edge)
    return added, len(edges) - before


def extract_doc(path: Path, catalog: dict[str, dict], model: str, use_cache: bool = False) -> dict:
    meta, _ = parse_doc(path)
    pid = meta["doc_id"]
    rels = "\n".join(f"- {r}: " + ", ".join(f"{a} -> {b}" for a, b in pairs) for r, pairs in REL_TYPES.items()
                     if r not in ("OFFERED_BY",))
    terms = ", ".join(f"{i} ({catalog[i]['name']})" for i in DOC_TYPE_TERMS[meta["doc_type"]])
    events = ", ".join(f"{i} ({catalog[i]['name']})" for i in DOC_TYPE_EVENTS[meta["doc_type"]])

    nodes, edges, failed = [], [], []
    for chunk in chunk_doc(path):
        previous = ", ".join(f"{n['id']} ({n['name']})" for n in nodes) or "нет"
        prompt = PROMPT.format(rels=rels, product_id=pid, product=meta["product"], terms=terms, events=events,
                               previous=previous, text=chunk["text"])
        cache = RAW_DIR_LLM / f"{chunk['id'].replace('#', '_')}.{_slug(model)}.json"
        try:
            if cache.exists() and use_cache:
                raw = cache.read_text(encoding="utf-8")
            else:
                raw = llm.chat([{"role": "user", "content": prompt}], model=model, schema=SCHEMA, max_tokens=MAX_TOKENS)
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(raw, encoding="utf-8")
            data = json.loads(raw)
            data["nodes"], data["edges"]  # noqa: B018 — проверка структуры
        except (llm.LLMUnavailable, json.JSONDecodeError, KeyError, TypeError) as e:
            # Обрезанный по MAX_TOKENS или битый JSON: раздел пропускаем, прогон продолжается.
            failed.append(chunk["id"])
            print(f"  {chunk['id']}: ПРОПУЩЕН ({e.__class__.__name__}: {str(e)[:80]})")
            continue
        added, n_edges = _merge(data, pid, catalog, nodes, edges)
        print(f"  {chunk['id']}: узлов {added} (от модели {len(data['nodes'])}), связей {n_edges}")
    return {"nodes": nodes, "edges": edges, "failed_chunks": failed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=llm.DEFAULT_MODEL)
    parser.add_argument("--mode", choices=["free", "classify"], default="free",
                        help="free — модель пишет узлы и связи сама; classify — выбирает из перечислений по пунктам")
    parser.add_argument("--only", nargs="*", help="doc_id одного или нескольких документов")
    parser.add_argument("--force", action="store_true", help="переизвлечь и уже готовые продукты")
    parser.add_argument("--reuse-raw", action="store_true",
                        help="не звать модель, если сырой ответ раздела уже сохранён (пересчёт постобработки)")
    args = parser.parse_args()
    if not llm.available() and not args.reuse_raw:
        parser.error(f"Ollama недоступен по {llm.OLLAMA_URL}")

    out_dir = CLASSIFY_OUT_DIR if args.mode == "classify" else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(GOLD_DIR / "glossary.json", out_dir / "glossary.json")
    glossary = json.loads((GOLD_DIR / "glossary.json").read_text(encoding="utf-8"))
    catalog = {n["id"]: n for n in glossary["nodes"]}

    failed_total = []
    for path in all_docs():
        meta, _ = parse_doc(path)
        if meta["doc_type"] in SKIP_TYPES or (args.only and meta["doc_id"] not in args.only):
            continue
        out = out_dir / f"{meta['doc_id']}.json"
        if out.exists() and not args.force and not args.only:
            print(f"{meta['doc_id']}: уже извлечён, пропускаю (--force — заново)")
            continue
        print(f"{meta['doc_id']} ({args.model}, {args.mode})", flush=True)
        started = time.time()
        if args.mode == "classify":
            classifier = ClauseClassifier(catalog, DOC_TYPE_TERMS[meta["doc_type"]], DOC_TYPE_EVENTS[meta["doc_type"]],
                                          args.model, RAW_DIR_LLM / "classify", use_cache=args.reuse_raw)
            result, stats = classifier.classify_doc(path)
            result["failed_chunks"] = stats.failed
            print(f"  отброшено связей: {len(stats.dropped_edges)}, значений вне перечисления: {stats.invalid_values}")
        else:
            result = extract_doc(path, catalog, args.model, use_cache=args.reuse_raw)
        result["seconds"] = round(time.time() - started, 1)
        print(f"  время: {result['seconds']} с", flush=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        failed_total += result["failed_chunks"]
    if failed_total:
        print(f"Пропущено разделов: {len(failed_total)} — {', '.join(failed_total)}")


if __name__ == "__main__":
    main()
