"""Маршрутизатор: Qwen3 со structured output выбирает rag / tool / both / refuse,
код проверяет и дополняет аргументы инструмента.

Модель 7–8B ненадёжна в function calling, поэтому:
- аргументы — плоские поля JSON-схемы, а не свободный объект;
- числа должны встречаться в вопросе (иначе это выдумка модели) — иначе уточняющий вопрос;
- параметры льготного периода берутся из каталога продукта, а не у модели;
- непонятный ответ модели — откат на rag.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from rag import llm
from rag.entities import Entities

CATALOG = Path(__file__).resolve().parent.parent / "data" / "catalog.json"
TOOLS = ["loan_calculator", "grace_period_calc", "compare_products", "save_summary"]

SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": ["rag", "tool", "both", "refuse"]},
        "tool": {"type": "string", "enum": ["none"] + TOOLS},
        "sum": {"type": ["number", "null"]},
        "rate": {"type": ["number", "null"]},
        "months": {"type": ["integer", "null"]},
        "loan_type": {"type": ["string", "null"], "enum": ["annuity", "diff", None]},
        "purchase_date": {"type": ["string", "null"]},
        "title": {"type": ["string", "null"]},
        "product_hints": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["route", "tool", "product_hints"],
}

PROMPT = """Ты маршрутизатор помощника, который объясняет условия банковских карт, кредитов и рассрочек.
Верни JSON по схеме.

route:
- rag — вопрос об условиях продуктов, терминах, последствиях, сравнение условий;
- tool — нужен только расчёт, продукт не важен;
- both — нужен расчёт и условия конкретного продукта, или сохранение разбора в файл;
- refuse — вопрос не о банковских продуктах (погода, рецепты, политика и т.п.).

tool:
- none — расчёт не нужен;
- loan_calculator — ежемесячный платёж и переплата: sum (₽), rate (% годовых), months, loan_type (annuity — равные платежи по умолчанию, diff — дифференцированные);
- grace_period_calc — до какой даты вернуть деньги без процентов: purchase_date (ГГГГ-ММ-ДД);
- compare_products — сравнить 2 и более продуктов;
- save_summary — сохранить разбор в файл: title.

Правила: числа бери только из вопроса; годы переводи в месяцы; для рассрочки без указанной ставки rate = 0;
если данных не хватает — оставь поле null. product_hints — названия продуктов из вопроса.
Сегодня {today}.
Продукты: {products}

Вопрос: {question}"""


# Расчёты и сравнения делает код, а не LLM. Qwen3 8B на прогоне выбирала rag для явных
# «сравни X и Y» и «сколько дней грейса у карты», поэтому такие вопросы распознаём сами.
COMPARE_RE = re.compile(r"\bсравн", re.IGNORECASE)
GRACE_RE = re.compile(r"до какой даты|сколько дней|когда (нужно )?вернуть|когда погасить", re.IGNORECASE)
GRACE_TERMS = {"term_grace", "term_grace_extension"}


@dataclass
class Route:
    route: str
    tool: str | None = None
    args: dict = field(default_factory=dict)
    product_ids: list[str] = field(default_factory=list)
    clarify: str | None = None
    note: str = ""


def question_numbers(question: str) -> set[float]:
    out = set()
    for m in re.finditer(r"\d[\d\s]*(?:[.,]\d+)?", question):
        raw = re.sub(r"\s+", "", m.group()).replace(",", ".")
        try:
            out.add(float(raw))
        except ValueError:
            continue
    return out


class Router:
    def __init__(self, chat=None, catalog_path: Path = CATALOG, today: date | None = None):
        # llm.chat берём в момент вызова: так его можно подменить в тестах интерфейса
        self.chat = chat or (lambda *a, **k: llm.chat(*a, **k))
        cat = json.loads(catalog_path.read_text(encoding="utf-8"))
        self.products = {p["id"]: p for p in cat["products"]}
        self.today = today or date.today()

    def ask_llm(self, question: str) -> dict:
        products = "; ".join(f"{p['name']} ({p['bank']})" for p in self.products.values())
        prompt = PROMPT.format(today=self.today.isoformat(), products=products, question=question)
        raw = self.chat([{"role": "user", "content": prompt}], schema=SCHEMA)
        return json.loads(raw)

    def route(self, question: str, ent: Entities) -> Route:
        try:
            data = self.ask_llm(question)
            if data.get("route") not in ("rag", "tool", "both", "refuse"):
                raise ValueError(data)
        except (ValueError, KeyError, TypeError) as e:  # кривой JSON — при сомнении rag
            return Route("rag", note=f"ответ маршрутизатора не разобран ({e.__class__.__name__}), откат на rag")
        return self.validate(data, question, ent)

    def validate(self, data: dict, question: str, ent: Entities) -> Route:
        route, tool = data["route"], data.get("tool") or "none"
        pids = list(ent.products)

        if route == "refuse":
            if ent.is_empty() and not ent.numbers:
                return Route("refuse")
            # В вопросе есть продукты, термины или числа — отказ сомнителен.
            route, tool = "rag", "none"
        note = ""
        if route != "refuse" and tool in ("none", None) and len(pids) >= 2 and COMPARE_RE.search(question):
            tool, note = "compare_products", "сравнение распознано кодом"
        elif (route != "refuse" and tool in ("none", None) and GRACE_TERMS & set(ent.content_terms)
              and GRACE_RE.search(question) and sum(bool(self.products[p].get("tool_params")) for p in pids) == 1):
            tool, note = "grace_period_calc", "расчёт льготного периода распознан кодом"
        if note:
            r = self.validate({**data, "route": "both", "tool": tool}, question, ent)
            r.note = "; ".join(x for x in (r.note, note) if x)
            return r
        if tool == "none" or tool not in TOOLS:
            return Route("rag" if route != "refuse" else "refuse", product_ids=pids)
        if route == "rag":
            route = "both"
        if pids and route == "tool":
            route = "both"  # продукт назван — к расчёту нужны его условия и дата редакции

        nums = question_numbers(question)
        if tool == "loan_calculator":
            return self._loan(data, nums, pids, route)
        if tool == "grace_period_calc":
            return self._grace(data, pids, route)
        if tool == "compare_products":
            if len(pids) < 2:
                return Route(route, tool, clarify="Какие продукты сравнить? Назовите хотя бы два, например «Опорная 55» и «Меридиан 150».")
            if len(pids) > 5:  # «Кедр и Балтийский» разворачивается в 8 продуктов: не обрезаем молча
                return Route(route, tool, clarify="Продуктов получилось больше пяти. Уточните, какие сравнить, "
                                                  "например «Кедр 78» и «100 дней».")
            return Route(route, tool, {"ids": pids}, pids)
        title = data.get("title") or ("Разбор: " + ", ".join(self.products[p]["name"] for p in pids) if pids else "Разбор")
        return Route("both", "save_summary", {"title": title}, pids)

    def _loan(self, data: dict, nums: set[float], pids: list[str], route: str) -> Route:
        s, rate, months = data.get("sum"), data.get("rate"), data.get("months")
        installment = any(self.products[p]["doc_type"] == "installment" for p in pids)
        if rate is None and installment:
            rate = 0
        missing = [name for name, v in (("сумму", s), ("ставку", rate), ("срок в месяцах", months)) if v is None]
        if missing:
            return Route(route, "loan_calculator", clarify=f"Чтобы посчитать, уточните {', '.join(missing)}.")
        if s not in nums:
            return Route(route, "loan_calculator", clarify=f"Уточните сумму: в вопросе я не нашёл {s:g} ₽.")
        if rate not in nums and not (rate == 0 and installment):
            return Route(route, "loan_calculator", clarify="Уточните процентную ставку (% годовых).")
        if months not in nums and months / 12 not in nums:
            return Route(route, "loan_calculator", clarify="Уточните срок в месяцах.")
        args = {"sum": float(s), "rate": float(rate), "months": int(months), "type": data.get("loan_type") or "annuity"}
        return Route(route, "loan_calculator", args, pids)

    def _grace(self, data: dict, pids: list[str], route: str) -> Route:
        cards = [p for p in pids if self.products[p].get("tool_params")]
        if len(cards) != 1:
            return Route(route, "grace_period_calc",
                         clarify="По какой кредитной карте посчитать? Например, «Полярная 120» или «Кедр 78».")
        params = self.products[cards[0]]["tool_params"]
        note = ""
        if not data.get("purchase_date"):
            # «Сколько дней грейса у карты?» — дата не важна, считаем от сегодняшнего дня.
            data["purchase_date"], note = self.today.isoformat(), "дата покупки не указана, считаю от сегодняшнего дня"
        args = {"purchase_date": data["purchase_date"], "grace_days": params["grace_days"], "mode": params["grace_mode"]}
        return Route(route, "grace_period_calc", args, pids, note=note)
