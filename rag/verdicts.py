"""Детерминированные выводы для вопросов-списков: «продукт × событие → вывод».

Вопросы вида «какие кредитные карты сохраняют льготный период при снятии наличных»,
«по каким рассрочкам при просрочке начисляются проценты», «какие студенческие карты
станут платными» модель 8B решала плохо: путала льготный период с бесплатным лимитом
комиссии, переносила условия одного продукта на другой, придумывала ставки.
Здесь вывод по каждому продукту строит код только из явных фактов графа,
а модель потом лишь пересказывает готовую таблицу.

Два вида вопросов:
- coverage — «распространяется ли условие T при событии E» (сейчас T = льготный период):
  CANCELS → «нет, отменяется»; исключение по E → «нет, не распространяется»;
  исключение по превышению лимита → «только в пределах лимита»;
  явный scope условия → «да»; иначе «нет данных» (не «да»).
- consequences — «что будет при событии E»: все TRIGGERS/CANCELS от E (и эскалации:
  пропуск платежа → пропуск двух) с пунктом; вывод по спрошенному термину:
  ставка → «да, проценты после …»; обслуживание → платно ли (по сумме «N ₽ в месяц»).
"""

import re
from dataclasses import dataclass, field

from rag.entities import Entities
from rag.graph_retriever import RELATED_EVENTS, Fact, GraphRetriever

COVERAGE_TERMS = {"term_grace"}
COVERAGE_EVENTS = {"ev_cash", "ev_transfer"}
PAID_RE = re.compile(r"(\d[\d\s]*)\s*₽\s*в месяц")


@dataclass
class Row:
    product: str
    product_id: str
    verdict: str  # «да», «нет, …», «только в пределах лимита», «нет данных», …
    detail: str
    sources: list[dict] = field(default_factory=list)  # {file, section, date}
    facts: list[Fact] = field(default_factory=list)  # из чего выведена строка (для DOT)

    def cite(self) -> str:
        seen = dict.fromkeys((s["file"], s["section"], s["date"]) for s in self.sources)
        return "; ".join(f"{f}, п. {sec}, ред. {d}" for f, sec, d in seen)


@dataclass
class Verdicts:
    kind: str  # coverage | consequences
    question_label: str
    rows: list[Row]

    def markdown(self) -> str:
        lines = [f"| Продукт | {self.question_label} | Подробности | Пункт и редакция |", "|---|---|---|---|"]
        lines += [f"| {r.product} | {r.verdict} | {r.detail} | {r.cite()} |" for r in self.rows]
        return "\n".join(lines)

    def products(self) -> set[str]:
        return {r.product for r in self.rows}

    def facts(self) -> list[Fact]:
        return [f for r in self.rows for f in r.facts]

    def summary(self) -> str:
        """Резюме, собранное кодом: запасной вариант, если резюме модели не прошло проверку."""
        groups: dict[str, list[str]] = {}
        for r in self.rows:
            groups.setdefault(r.verdict.split(",")[0], []).append(f"«{r.product}»")
        return " ".join(f"{k.capitalize()}: {', '.join(v)}." for k, v in groups.items())


def detect(ent: Entities) -> str | None:
    """Вопрос-список: продукт не назван, назван тип продукта и событие."""
    if ent.products or not ent.type_terms or not ent.events:
        return None
    if COVERAGE_TERMS & set(ent.content_terms) and COVERAGE_EVENTS & set(ent.events):
        return "coverage"
    return "consequences"


def _after(text: str) -> str:
    return text.split(" — ", 1)[1] if " — " in text else text


class VerdictBuilder:
    def __init__(self, graph: GraphRetriever):
        self.g = graph
        self.event_names = {i: n for i, n, *_ in graph.store.catalog("Event")}

    def build(self, ent: Entities) -> Verdicts | None:
        kind = detect(ent)
        if kind is None:
            return None
        pids = sorted(p for p, t in self.g.product_type.items() if t in ent.type_terms)
        if kind == "coverage":
            events = COVERAGE_EVENTS & set(ent.events)
            label = "Льготный период при: " + " и ".join(sorted(self.event_names[e].lower() for e in events))
            return Verdicts(kind, label, [self._coverage(pid, events) for pid in pids])
        events = set(ent.events) | {r for e in ent.events for r in RELATED_EVENTS.get(e, [])}
        return Verdicts(kind, self._consequence_label(ent), [self._consequences(pid, events, ent) for pid in pids])

    def _src(self, f: Fact) -> dict:
        return self.g.source(f.product_id if f.rel_section else f.node_id, f.rel_section)

    def _row(self, pid: str, verdict: str, detail: str, facts: list[Fact]) -> Row:
        return Row(self.g.products[pid], pid, verdict, detail, [self._src(f) for f in facts], facts)

    def _coverage(self, pid: str, events: set[str]) -> Row:
        facts = self.g.product_facts(pid)
        grace = [f for f in facts if f.kind == "condition" and f.term_id in COVERAGE_TERMS]
        cancels = [f for f in facts if f.kind == "cancels" and f.term_id in COVERAGE_TERMS and f.event_id in events]
        excl = [f for f in facts if f.kind == "exception" and f.term_id in COVERAGE_TERMS and f.event_id in events]
        limit = [f for f in facts if f.kind == "exception" and f.term_id in COVERAGE_TERMS
                 and f.event_id == "ev_limit_exceeded"]
        scopes = self.g.store.condition_scopes(pid)
        covered = [f for f in grace if events <= scopes.get(f.node_id, set())]
        fees = [f for f in facts if f.kind == "triggers" and f.event_id in events]

        if cancels:
            return self._row(pid, "нет, отменяется", "; ".join(
                f"«{self.event_names[f.event_id]}» отменяет льготный период" for f in cancels), cancels)
        if excl:
            return self._row(pid, "нет, не распространяется", "; ".join(_after(f.text) for f in excl), excl)
        if limit:
            return self._row(pid, "только в пределах лимита", "; ".join(_after(f.text) for f in limit), limit + covered)
        if covered:
            detail = "; ".join(f.text.split(": ", 1)[1] for f in covered)
            if fees:
                detail += "; но комиссия: " + "; ".join(_after(f.text) for f in fees)
            return self._row(pid, "да", detail, covered + fees)
        return self._row(pid, "нет данных", "в условиях нет явного правила", [])

    @staticmethod
    def _consequence_label(ent: Entities) -> str:
        if "term_rate" in ent.content_terms:
            return "Начисляются ли проценты"
        if "term_service_fee" in ent.content_terms:
            return "Станет ли платной"
        return "Последствия"

    def _consequences(self, pid: str, events: set[str], ent: Entities) -> Row:
        facts = [f for f in self.g.product_facts(pid) if f.kind in ("triggers", "cancels") and f.event_id in events]
        order = {e: i for i, e in enumerate(sorted(events, key=lambda e: (e not in ent.events, e)))}
        facts.sort(key=lambda f: (order.get(f.event_id, 9), f.kind))
        if not facts:
            return self._row(pid, "нет данных", "в условиях нет последствий этого события", [])

        def short(f: Fact) -> str:
            sep = "» ВЛЕЧЁТ " if f.kind == "triggers" else "» ОТМЕНЯЕТ "
            verb = "влечёт" if f.kind == "triggers" else "отменяет"
            return f"{self.event_names[f.event_id]}: {verb} {f.text.split(sep, 1)[1]}"
        detail = "; ".join(short(f) for f in facts)

        if "term_rate" in ent.content_terms:
            rate = next((f for f in facts if f.kind == "triggers" and f.term_id == "term_rate"), None)
            verdict = (f"да, после «{self.event_names[rate.event_id].lower()}»" if rate
                       else "нет процентов, другие последствия")
        elif "term_service_fee" in ent.content_terms:
            fee = next((f for f in facts if f.kind == "triggers" and f.term_id == "term_service_fee"), None)
            amounts = [int(re.sub(r"\s", "", m)) for m in PAID_RE.findall(fee.text)] if fee else []
            verdict = ((f"да, {max(amounts)} ₽ в месяц, если условия не выполнены" if any(amounts)
                        else "нет, обслуживание 0 ₽") if fee else "нет данных")
        else:
            verdict = "есть последствия"
        return self._row(pid, verdict, detail, facts)
