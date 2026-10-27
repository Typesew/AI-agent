"""Графовый ретривер на эталонном графе: вопросы из раздела 6 ТЗ."""

import tempfile
import unittest
from pathlib import Path

from indexing import build_graph
from rag.graph_retriever import GraphRetriever, to_dot

GOLD = build_graph.ROOT / "data" / "extracted" / "gold"


class GraphTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        db = Path(cls.tmp.name) / "graph.kuzu"
        build_graph.build(GOLD, db)
        cls.g = GraphRetriever(db)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def facts(self, q, kind=None):
        return [f for f in self.g.retrieve(q).facts if kind is None or f.kind == kind]

    def test_q2_what_cancels_grace(self):
        facts = self.facts("Что отменяет льготный период по карте «Полярная 120»?")
        cancels = {f.event_id for f in facts if f.kind == "cancels"}
        self.assertEqual(cancels, {"ev_missed_payment", "ev_not_full_repay"})
        excluded = {f.event_id for f in facts if f.kind == "exception"}
        self.assertEqual(excluded, {"ev_cash", "ev_transfer"})
        self.assertTrue(all(f.source["file"] == "sever_credit.md" for f in facts if f.kind != "definition"))

    def test_q3_cashback_exceptions(self):
        texts = " ".join(f.text for f in self.facts("Какие операции не участвуют в кэшбэке по «Полярная 120»?", "exception"))
        for item in ("На все покупки", "Снятие наличных", "Переводы", "ЖКХ"):
            self.assertIn(item, texts)

    def test_q4_cross_product_grace_and_cash(self):
        facts = self.facts("Какие кредитные карты сохраняют льготный период при снятии наличных?")
        products = {f.product for f in facts if f.kind != "definition"}
        self.assertEqual(products, {"Полярная 120", "Опорная 55", "Меридиан 150", "Кедр 78", "100 дней"})
        text = " ".join(f.text for f in facts)
        self.assertIn("3,9% + 390 ₽", text)
        self.assertIn("сверх 50 000 ₽", text)
        opora = [f for f in facts if f.product == "Опорная 55" and f.kind == "exception"]
        self.assertEqual([f.event_id for f in opora], ["ev_limit_exceeded"])
        no_exceptions = [f for f in facts if f.product == "100 дней" and f.kind == "exception"]
        self.assertEqual(no_exceptions, [])

    def test_q5_installments_with_interest(self):
        facts = self.facts("По каким рассрочкам при просрочке начисляются проценты?")
        by_product = {f.product: f.text for f in facts if f.kind == "triggers"}
        self.assertEqual(set(by_product), {"Меридиан Частями", "Плати частями", "Балтийская Рассрочка",
                                           "Рассрочка по кнопке", "Кедр Рассрочка"})
        self.assertIn("Доплата", by_product["Кедр Рассрочка"])

    def test_q6_compare_table(self):
        r = self.g.retrieve("Сравни по обслуживанию и снятию наличных кредитки «Опорная 55» и «Меридиан 150».")
        self.assertIn("4,9% + 490 ₽", r.table)
        self.assertIn("5,9% + 590 ₽", r.table)
        self.assertIn("590 ₽ в месяц", r.table)  # подводный камень: окончание акции

    def test_q7_paid_service(self):
        text = " ".join(f.text for f in self.facts("Когда у «Опорная Black» платное обслуживание?"))
        self.assertIn("99 ₽ в месяц", text)
        self.assertIn("50 000 ₽", text)

    def test_q9_students_becoming_paid(self):
        facts = self.facts("Какие студенческие карты после потери статуса могут стать платными?", "triggers")
        paid = {f.product for f in facts if "99 ₽" in f.text or "149 ₽" in f.text}
        self.assertEqual(paid, {"Молодёжная Опорная", "Студенческая Балтийская"})
        self.assertEqual(len(facts), 5)

    def test_q14_missed_one_and_two(self):
        facts = self.facts("Объясни, что будет, если пропущу платёж по «Плати частями», и что при двух пропусках.")
        text = " ".join(f.text for f in facts)
        self.assertIn("500 ₽", text)
        self.assertIn("39,9%", text)
        self.assertTrue(all(f.source["section"].startswith("3.") for f in facts))

    def test_out_of_domain(self):
        self.assertEqual(self.facts("Какая погода завтра?"), [])

    def test_dot(self):
        dot = to_dot(self.facts("Что отменяет льготный период по карте «Полярная 120»?"))
        self.assertTrue(dot.startswith("digraph"))
        self.assertIn("отменяет", dot)  # подписи рёбер по-русски


if __name__ == "__main__":
    unittest.main()
