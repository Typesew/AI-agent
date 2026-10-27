"""Распознавание в вопросе банка, продукта, типа продукта, терминов и событий.

Каталог берётся из data/catalog.json (его пишет corpus.build_corpus), поэтому
распознавание работает и без графа.

Правила неоднозначности:
1. Названия продуктов и банков ищутся как подряд идущие слова; при пересечении
   побеждает самое длинное совпадение («Студенческая Северная карта» — это
   «Студенческая Северная», а не «Северная Карта»; «Меридиан 150» — продукт, а не банк).
2. Назван только банк и тип («рассрочка в Кедре») — берём продукт этого типа в банке;
   назван только банк — все его продукты.
3. Продукт с require_context («100 дней») засчитывается, только если в вопросе
   есть слово контекста («карта», «кредитка», «Балтийский»): «за 100 дней» — не продукт.
4. Слова из названий найденных продуктов не считаются терминами
   («Кедр Рассрочка» не делает вопрос вопросом про рассрочку вообще).
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from rag.text import contains_all, find_spans, norm, phrase_stems, tokens, word_matches

CATALOG = Path(__file__).resolve().parent.parent / "data" / "catalog.json"
NUMBER_RE = re.compile(r"\d[\d\s]*(?:[.,]\d+)?")
DEFINITION_RE = re.compile(r"что (такое|значит|означает)|что это")
# Слова предметной области помимо каталога: по ним отличаем «кредит 100 000 под 18%»
# (расчёт, не отказ) от «какая погода завтра».
DOMAIN_STEMS = ["кредит", "карт", "банк", "ставк", "плат", "процент", "рассроч", "вклад", "депозит", "займ",
                "заем", "долг", "комисс", "кэшбэк", "кешбэк", "ипотек", "переплат", "аннуитет", "сохрани",
                "разбор", "счет", "деньг", "руб", "₽", "переплат", "грейс", "льготн"]


@dataclass
class Entities:
    banks: dict[str, str] = field(default_factory=dict)
    products: dict[str, str] = field(default_factory=dict)
    type_terms: dict[str, str] = field(default_factory=dict)
    content_terms: dict[str, str] = field(default_factory=dict)
    events: dict[str, str] = field(default_factory=dict)
    numbers: list[str] = field(default_factory=list)
    domain: bool = False
    definition: bool = False  # «что такое …» — отвечаем по глоссарию и памяткам

    def is_empty(self) -> bool:
        return not (self.banks or self.products or self.type_terms or self.content_terms or self.events)

    def summary(self) -> dict[str, list[str]]:
        keys = ("banks", "products", "type_terms", "content_terms", "events")
        return {k: list(getattr(self, k).values()) for k in keys if getattr(self, k)}


class EntityRecognizer:
    def __init__(self, catalog_path: Path = CATALOG):
        cat = json.loads(catalog_path.read_text(encoding="utf-8"))
        self.products = {p["id"]: p for p in cat["products"]}
        self.banks = {b["id"]: b for b in cat["banks"]}
        self.terms = {t["id"]: t for t in cat["terms"]}
        self.events = {e["id"]: e for e in cat["events"]}
        self.product_types = set(cat["product_types"])

    @staticmethod
    def _phrases(entry: dict) -> list[str]:
        return [a for a in entry["aliases"].split("|") if a]

    def _named_spans(self, words: list[str]) -> list[tuple[int, int, str, str]]:
        """Кандидаты (start, end, kind, id) для продуктов и банков."""
        cands = []
        for pid, p in self.products.items():
            ctx = p.get("require_context") or []
            if ctx and not any(word_matches(c, w) for c in ctx for w in words):
                continue
            for phrase in self._phrases(p):
                cands += [(s, e, "product", pid) for s, e in find_spans(phrase, words)]
        for bid, b in self.banks.items():
            for phrase in self._phrases(b):
                cands += [(s, e, "bank", bid) for s, e in find_spans(phrase, words)]
        return cands

    def recognize(self, question: str) -> Entities:
        words = tokens(question)
        # Самое длинное совпадение побеждает: по числу слов, затем по длине текста.
        cands = sorted(self._named_spans(words), key=lambda c: (c[1] - c[0], len(" ".join(words[c[0]:c[1]]))),
                       reverse=True)
        taken: set[int] = set()
        ent = Entities()
        for s, e, kind, eid in cands:
            if taken & set(range(s, e)):
                continue
            taken |= set(range(s, e))
            if kind == "product":
                ent.products[eid] = self.products[eid]["name"]
            else:
                ent.banks[eid] = self.banks[eid]["name"]

        # Слова названий не участвуют в поиске терминов («Кедр Рассрочка» != «рассрочка»).
        name_stems = [s for pid in ent.products for p in self._phrases(self.products[pid]) for s in phrase_stems(p)]
        rest = [w for i, w in enumerate(words)
                if i not in taken and not any(word_matches(s, w) for s in name_stems)]

        for tid, t in self.terms.items():
            if any(contains_all(p, rest) for p in self._phrases(t)):
                (ent.type_terms if tid in self.product_types else ent.content_terms)[tid] = t["name"]
        for eid, ev in self.events.items():
            if any(contains_all(p, rest) for p in self._phrases(ev)):
                ent.events[eid] = ev["name"]

        if ent.banks and not ent.products:
            for pid, p in self.products.items():
                if p["bank_id"] in ent.banks and (not ent.type_terms or p["type_term"] in ent.type_terms):
                    ent.products[pid] = p["name"]

        ent.numbers = [n.strip() for n in NUMBER_RE.findall(question)]
        text = norm(question)
        ent.definition = bool(DEFINITION_RE.search(text))
        ent.domain = not ent.is_empty() or any(s in text for s in DOMAIN_STEMS)
        return ent
