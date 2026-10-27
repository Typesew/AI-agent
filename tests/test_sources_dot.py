"""Блок «Источники» (один документ — одна строка, без лишних продуктов) и DOT-схема графа."""

import unittest

from rag.graph_retriever import Fact, to_dot
from rag.prompt import Source, collect_sources


def items():
    return {
        "c1": ("", Source("balt_credit.md", "раздел 2. Льготный период", "2026-09-15", "«100 дней» (Банк «Балтийский»)")),
        "f1": ("", Source("balt_credit.md", "п. 2.1", "2026-09-15", "100 дней")),
        "f2": ("", Source("balt_credit.md", "п. 3.1", "2026-09-15", "100 дней")),
        "f3": ("", Source("kedr_credit.md", "п. 2.1", "2026-07-20", "Кедр 78")),
        "f4": ("", Source("opora_credit.md", "п. 2.2", "2026-08-15", "Опорная 55")),
        "g1": ("", Source("glossary.md", "п. 1.1", "2026-09-01", "Глоссарий")),
    }


class SourcesTest(unittest.TestCase):
    def test_one_line_per_document(self):
        out = collect_sources(list(items()), items())
        files = [s.file for s in out]
        self.assertEqual(len(files), len(set(files)))
        balt = next(s for s in out if s.file == "balt_credit.md")
        self.assertEqual(balt.section, "пп. 2.1, 3.1")
        self.assertIn("Балтийский", balt.title)  # берём самое полное название

    def test_products_not_mentioned_in_answer_are_dropped(self):
        answer = "Карты «100 дней» и «Опорной 55» сохраняют льготный период при снятии наличных."
        files = {s.file for s in collect_sources(list(items()), items(), answer)}
        self.assertEqual(files, {"balt_credit.md", "opora_credit.md", "glossary.md"})

    def test_no_product_named_means_no_filtering(self):
        files = {s.file for s in collect_sources(list(items()), items(), "Ответ без названий продуктов.")}
        self.assertIn("kedr_credit.md", files)


class DotTest(unittest.TestCase):
    def facts(self):
        return [
            Fact("100 дней", "condition", "", "c1", path=[("100 дней", "HAS_CONDITION", "Льготный период")], product_id="balt_credit"),
            Fact("Кедр 78", "exception", "", "x1", product_id="kedr_credit",
                 path=[("Кедр 78", "HAS_CONDITION", "Льготный период"), ("Льготный период", "HAS_EXCEPTION", "Снятие наличных")]),
        ]

    def test_same_condition_name_in_two_products_is_two_nodes(self):
        dot = to_dot(self.facts())
        self.assertEqual(dot.count("Льготный период"), 2)
        self.assertIn("(100 дней)", dot)
        self.assertIn("(Кедр 78)", dot)

    def test_edge_labels_are_russian(self):
        dot = to_dot(self.facts())
        self.assertIn("не распространяется на", dot)
        self.assertNotIn("HAS_EXCEPTION", dot)


if __name__ == "__main__":
    unittest.main()
