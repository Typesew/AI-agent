"""Обогащённый промпт и разбор ответа.

В промпт идут фрагменты [c1]…, факты графа [f1]… и результаты инструментов [t1]….
Модель не ставит ссылки в тексте, а в последней строке пишет «ИСПОЛЬЗОВАНО: c1,f3»;
код вырезает строку и сам собирает единый блок «Источники» только по этим id.
Если строки нет или id неверные — показываем все источники контекста.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rag.retriever import Context
from rag.text import find_spans, tokens

SYSTEM = """Ты «Банковский переводчик»: объясняешь условия банковских карт, кредитов и рассрочек простым языком.

Правила:
1. Отвечай только по контексту. Не придумывай ставки, суммы, сроки и даты.
2. Числа расчётов бери только из результатов инструментов [t…], сам ничего не считай.
3. Если в контексте нет ответа, напиши ровно: «В корпусе нет данных по этому вопросу.»
4. Структура ответа — разделы «**Плюсы**», «**Минусы**», «**Подводные камни**», «**Итог**».
   Для вопроса-определения или чистого расчёта можно короче: только нужные разделы, пустых не пиши.
5. Если вопрос про то, какие продукты подходят («какие карты», «по каким рассрочкам»), назови КАЖДЫЙ продукт
   из контекста и для каждого скажи прямо: да, нет или с оговоркой. Слова «большинство» и «многие» без перечисления не пиши.
6. Не противоречь себе: если у продукта есть условие или лимит (например, «в пределах 50 000 ₽ в месяц»),
   укажи его там, где говоришь о плюсе или минусе, а не в разных разделах разными словами.
7. Не ставь ссылки в тексте. Последней строкой напиши id только тех фрагментов, на которые реально опирался ответ
   (обычно 2–8), а не всего контекста:
   ИСПОЛЬЗОВАНО: c1,f2,t1"""

USED_RE = re.compile(r"^\s*ИСПОЛЬЗОВАНО\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE)


@dataclass
class Source:
    file: str
    section: str
    date: str
    title: str

    def render(self) -> str:
        return f"{self.title} — {self.file}, {self.section}, ред. {self.date}"


def context_items(ctx: Context, tool_results: list[dict] | None = None) -> dict[str, tuple[str, Source | None]]:
    """id -> (текст для промпта, источник)."""
    items: dict[str, tuple[str, Source | None]] = {}
    for i, c in enumerate(ctx.chunks, 1):
        m = c.metadata
        title = f"«{m['product']}» ({m['bank']})" if m.get("product", "-") != "-" else c.text.splitlines()[0]
        items[f"c{i}"] = (c.text, Source(m["source_file"], f"раздел {m['section']}", m["date"], title))
    for i, f in enumerate(ctx.facts, 1):
        src = f.source
        items[f"f{i}"] = (f.text, Source(src["file"], f"п. {src['section']}", src["date"], f.product) if src else None)
    for i, t in enumerate(tool_results or [], 1):
        items[f"t{i}"] = (f"{t['tool']}({json.dumps(t['args'], ensure_ascii=False)}) = "
                          f"{json.dumps(t['result'], ensure_ascii=False)}", None)
    return items


def build_messages(question: str, ctx: Context, tool_results: list[dict] | None = None) -> list[dict]:
    items = context_items(ctx, tool_results)
    body = "\n\n".join(f"[{k}] {text}" for k, (text, _) in items.items()) or "(контекст пуст)"
    if ctx.table:
        body += "\n\nСравнение из графа:\n" + ctx.table
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Контекст:\n{body}\n\nВопрос: {question}"}]


def parse_answer(text: str, items: dict) -> tuple[str, list[str], bool]:
    """(ответ без служебной строки, использованные id, удалось ли их разобрать)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    matches = list(USED_RE.finditer(text))
    if not matches:
        return text, list(items), False
    last = matches[-1]
    body = (text[:last.start()] + text[last.end():]).strip()
    ids = [i.strip().strip("[]").lower() for i in re.split(r"[,\s;]+", last.group(1)) if i.strip()]
    valid = [i for i in ids if i in items]
    if not valid:
        return body, list(items), False
    return body, valid, True


def _catalog_aliases() -> dict[str, list[str]]:
    """doc_id продукта -> названия, по которым его можно узнать в тексте ответа."""
    path = Path(__file__).resolve().parent.parent / "data" / "catalog.json"
    if not path.exists():
        return {}
    cat = json.loads(path.read_text(encoding="utf-8"))
    return {p["id"]: [a for a in p["aliases"].split("|") if a] for p in cat["products"]}


def _mentioned(answer: str, aliases: list[str]) -> bool:
    words = tokens(answer)
    return any(find_spans(a, words) for a in aliases)


def collect_sources(used: list[str], items: dict, answer: str = "") -> list[Source]:
    """Один источник на документ: пункты склеены, продукты без упоминания в ответе отброшены.

    Модель 7–8B любит перечислять в «ИСПОЛЬЗОВАНО» весь контекст, поэтому код дополнительно
    оставляет документы продуктов, которые названы в тексте ответа (общие документы —
    глоссарий, памятки, разборы — остаются по строке «ИСПОЛЬЗОВАНО»). Если в ответе
    не назван ни один из использованных продуктов, показываем список без фильтра.
    """
    docs: dict[str, dict] = {}
    for i in used:
        src = items.get(i, (None, None))[1]
        if not src:
            continue
        d = docs.setdefault(src.file, {"date": src.date, "title": src.title, "clauses": [], "sections": []})
        if len(src.title) > len(d["title"]):
            d["title"] = src.title  # «100 дней» (Банк «Балтийский») лучше, чем просто «100 дней»
        part = src.section
        if part.startswith("п. "):
            if part[3:] not in d["clauses"]:
                d["clauses"].append(part[3:])
        elif part not in d["sections"]:
            d["sections"].append(part)

    def build(selected: dict[str, dict]) -> list[Source]:
        out = []
        for file, d in selected.items():
            if d["clauses"]:  # конкретные пункты точнее, чем «раздел 2. Льготный период»
                section = ("пп. " if len(d["clauses"]) > 1 else "п. ") + ", ".join(d["clauses"])
            else:
                section = "; ".join(d["sections"])
            out.append(Source(file, section, d["date"], d["title"]))
        return out

    aliases = _catalog_aliases()
    if answer and aliases:
        product_docs = [f for f in docs if Path(f).stem in aliases]
        kept = {f: d for f, d in docs.items() if f not in product_docs or _mentioned(answer, aliases[Path(f).stem])}
        # Если ни один продукт в ответе не назван, фильтр ничего не доказывает: оставляем всё как есть.
        if any(f in kept for f in product_docs):
            return build(kept)
    return build(docs)
