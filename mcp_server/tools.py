"""Детерминированные инструменты агента. Считает код, а не LLM.

Функции чистые (кроме compare_products — чтение графа, и save_summary — запись
файла) и тестируются без MCP; mcp_server.server только публикует их.
"""

import json
import math
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUMMARY_DIR = Path(os.environ.get("BANK_SUMMARY_DIR", ROOT / "data" / "summaries"))
CATALOG = ROOT / "data" / "catalog.json"
GRAPH_DB = Path(os.environ.get("BANK_GRAPH_DB", ROOT / "data" / "graph.kuzu"))

# Параметры продления льготного периода «Меридиан 150» (merid_credit.md, п. 2.1–2.2)
BASE_GRACE_DAYS, EXTENSION_DAYS = 60, 90
EXTENSION_COST = "первое продление бесплатно, каждое следующее — 1,9% от суммы долга на начало месяца"
COMPARE_TERMS = ["term_service_fee", "term_cash_withdrawal", "term_grace", "term_cashback"]


class ToolError(ValueError):
    """Неверные аргументы: сообщение показывается пользователю как есть."""


def _number(name: str, value, lo: float, hi: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(value):
        raise ToolError(f"{name} должен быть числом, получено {value!r}")
    if not lo <= value <= hi:
        raise ToolError(f"{name} должен быть от {lo:g} до {hi:g}, получено {value:g}")
    return float(value)


def _money(x: float) -> float:
    return round(x + 1e-9, 2)


def loan_calculator(sum: float, rate: float, months: int, type: str = "annuity", fee: float = 0.0) -> dict:
    """Платёж и переплата по кредиту или рассрочке.

    sum — сумма, ₽; rate — годовая ставка в процентах (0 для рассрочки); months — срок;
    type — annuity (равные платежи) или diff (дифференцированные); fee — разовая комиссия, ₽.
    Аннуитет: P = S·r / (1 − (1+r)^(−n)), r = rate/12/100; при rate = 0 платёж S/n.
    Дифференцированный: переплата S·r·(n+1)/2.
    """
    s = _number("sum", sum, 1, 1e9)
    rate_ = _number("rate", rate, 0, 100)
    if isinstance(months, float) and months.is_integer():
        months = int(months)
    if not isinstance(months, int) or isinstance(months, bool) or not 1 <= months <= 600:
        raise ToolError(f"months должен быть целым от 1 до 600, получено {months!r}")
    if type not in ("annuity", "diff"):
        raise ToolError(f"type должен быть annuity или diff, получено {type!r}")
    fee_ = _number("fee", fee, 0, 1e9)
    r, n = rate_ / 100 / 12, months
    result = {"type": type, "sum": s, "rate": rate_, "months": n, "fee": fee_}

    if type == "annuity":
        payment = s / n if r == 0 else s * r / (1 - (1 + r) ** -n)
        interest = payment * n - s
        result["monthly_payment"] = _money(payment)
    else:
        principal = s / n
        schedule = [principal + (s - principal * i) * r for i in range(n)]
        interest = s * r * (n + 1) / 2
        result["first_payment"] = _money(schedule[0])
        result["last_payment"] = _money(schedule[-1])
        result["schedule"] = [_money(p) for p in schedule]
    result["total"] = _money(s + interest + fee_)
    result["overpayment"] = _money(interest + fee_)
    return result


def _parse_date(value: str) -> date:
    for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(str(value).strip(), fmt).date()
        except ValueError:
            continue
    raise ToolError(f"purchase_date: ожидается дата ГГГГ-ММ-ДД или ДД.ММ.ГГГГ, получено {value!r}")


def _next_statement(after: date, day: int) -> date:
    y, m = after.year, after.month
    for _ in range(2):
        last = (date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)).day
        d = date(y, m, min(day, last))
        if d >= after:
            return d
        y, m = y + (m == 12), m % 12 + 1
    raise AssertionError("недостижимо")


def grace_period_calc(purchase_date: str, statement_day: int = 0, grace_days: int = 0,
                      mode: str = "from_purchase") -> dict:
    """До какой даты вернуть долг без процентов.

    Соглашение: период из N дней начинается в дату старта включительно,
    последний беспроцентный день = старт + N − 1. Вернуть долг нужно не позже него.

    mode:
    - from_purchase — старт в дату покупки;
    - from_month_start — старт 1-го числа месяца покупки («Полярная 120»);
    - base_plus_extension — старт в дату покупки, 60 базовых + 90 дней продления
      («Меридиан 150»), grace_days игнорируется, в ответ добавляется стоимость продления.
    statement_day (1–31, 0 — не задан) — день выписки; если задан, в ответе будет ближайшая выписка.
    """
    purchase = _parse_date(purchase_date)
    if mode not in ("from_purchase", "from_month_start", "base_plus_extension"):
        raise ToolError(f"mode должен быть from_purchase, from_month_start или base_plus_extension, получено {mode!r}")
    if not isinstance(statement_day, int) or not 0 <= statement_day <= 31:
        raise ToolError(f"statement_day должен быть целым от 0 до 31, получено {statement_day!r}")

    if mode == "base_plus_extension":
        grace_days = BASE_GRACE_DAYS + EXTENSION_DAYS
    elif not isinstance(grace_days, int) or isinstance(grace_days, bool) or not 1 <= grace_days <= 365:
        raise ToolError(f"grace_days должен быть целым от 1 до 365, получено {grace_days!r}")

    start = purchase.replace(day=1) if mode == "from_month_start" else purchase
    last = start + timedelta(days=grace_days - 1)
    result = {
        "mode": mode, "purchase_date": purchase.isoformat(), "start": start.isoformat(),
        "grace_days": grace_days, "last_day": last.isoformat(),
        "days_from_purchase": (last - purchase).days + 1,
    }
    if mode == "base_plus_extension":
        result["base_days"], result["extension_days"] = BASE_GRACE_DAYS, EXTENSION_DAYS
        result["base_last_day"] = (start + timedelta(days=BASE_GRACE_DAYS - 1)).isoformat()
        result["extension_cost"] = EXTENSION_COST
    if statement_day:
        result["next_statement"] = _next_statement(purchase, statement_day).isoformat()
    return result


def compare_products(ids: list[str]) -> dict:
    """Сводная таблица условий из графа по общим терминам: обслуживание, снятие наличных, грейс, кэшбэк."""
    from rag.graph_retriever import GraphRetriever  # граф нужен только этому инструменту

    if not isinstance(ids, list) or not 2 <= len(ids) <= 5:
        raise ToolError("ids: нужен список из 2–5 идентификаторов продуктов")
    known = {p["id"]: p["name"] for p in json.loads(CATALOG.read_text(encoding="utf-8"))["products"]}
    unknown = [i for i in ids if i not in known]
    if unknown:
        raise ToolError(f"неизвестные продукты: {', '.join(map(str, unknown))}; допустимые: {', '.join(sorted(known))}")
    if not GRAPH_DB.exists():
        raise ToolError("граф не построен: python -m indexing.build_graph")

    g = GraphRetriever(GRAPH_DB)
    rows: dict[str, dict[str, list[str]]] = {}
    sources, seen = set(), set()
    for pid in ids:
        for f in g.product_facts(pid):
            if f.node_id in seen:  # комиссия, которая ещё и последствие события, — одна строка
                continue
            if f.term_id in COMPARE_TERMS and f.kind in ("condition", "fee", "exception"):
                seen.add(f.node_id)
                rows.setdefault(f.term_id, {}).setdefault(known[pid], []).append(f.text.split(": ", 1)[1])
                src = g.source(f.product_id if f.rel_section else f.node_id, f.rel_section)
                sources.add((src["file"], src["section"], src["date"]))
    names = [known[i] for i in ids]
    lines = ["| Параметр | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for tid in COMPARE_TERMS:
        if tid in rows:
            label = g.terms[tid]["name"]
            lines.append(f"| {label} | " + " | ".join("; ".join(rows[tid].get(n, ["нет данных"])) for n in names) + " |")
    return {"products": names, "table": "\n".join(lines),
            "sources": [{"file": f, "section": s, "date": d} for f, s, d in sorted(sources)]}


def save_summary(title: str, text: str) -> dict:
    """Сохраняет разбор в data/summaries/<имя>.md. Имя санитизируется, выйти из папки нельзя."""
    if not isinstance(title, str) or not title.strip():
        raise ToolError("title: нужна непустая строка")
    if not isinstance(text, str) or not text.strip():
        raise ToolError("text: нужна непустая строка")
    if len(text) > 100_000:
        raise ToolError("text: не больше 100 000 символов")
    name = re.sub(r"[^\w\-]+", "_", title.strip(), flags=re.UNICODE).strip("_.")[:60] or "summary"
    folder = SUMMARY_DIR.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    path = (folder / f"{name}.md").resolve()
    i = 2
    while path.exists():
        path = (folder / f"{name}-{i}.md").resolve()
        i += 1
    if path.parent != folder:
        raise ToolError("недопустимое имя файла")
    path.write_text(f"# {title.strip()}\n\n{text.strip()}\n", encoding="utf-8")
    return {"path": str(path.relative_to(ROOT)) if ROOT in path.parents else str(path), "bytes": path.stat().st_size}
