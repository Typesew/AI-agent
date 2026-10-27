"""Этап B: выводы для вопросов-списков строит код по графу.

Ожидаемые значения считаются прямо из YAML продуктов — независимо от ретривера и графа.
"""

import json
import re
import tempfile
import unittest
from pathlib import Path

import yaml

from agent.agent import Agent
from agent.router import Router
from indexing import build_graph
from rag.entities import EntityRecognizer
from rag.graph_retriever import GraphRetriever
from rag.prompt import build_list_messages, summary_violations
from rag.verdicts import VerdictBuilder

PRODUCTS = Path(__file__).resolve().parent.parent / "corpus" / "products"
GLOSSARY = yaml.safe_load((PRODUCTS.parent / "glossary.yaml").read_text(encoding="utf-8"))
EVENT_NAMES = {e["id"]: e["name"] for e in GLOSSARY["events"]}
Q4 = "Какие кредитные карты сохраняют льготный период при снятии наличных?"
Q5 = "По каким рассрочкам при просрочке начисляются проценты?"
Q9 = "Какие студенческие карты после потери статуса могут стать платными?"


def yaml_products(doc_type: str) -> list[dict]:
    return [p for p in (yaml.safe_load(f.read_text(encoding="utf-8")) for f in sorted(PRODUCTS.glob("*.yaml")))
            if p["doc_type"] == doc_type]


def yaml_nodes(p: dict) -> list[dict]:
    return [n for s in p["sections"] for c in s["clauses"] for n in c.get("nodes", []) + ([c["node"]] if "node" in c else [])]


def expected_q4() -> dict[str, str]:
    out = {}
    for p in yaml_products("credit_card"):
        nodes = yaml_nodes(p)
        grace = next(n["id"] for n in nodes if n.get("term") == "term_grace")
        exc = {n.get("event") for n in nodes if n["label"] == "Exception" and n.get("of") == grace}
        cancels = any(e["rel"] == "CANCELS" and e["event"] == "ev_cash" and e["target"] == grace for e in p["events"])
        scope = next(n.get("scope", []) for n in nodes if n["id"] == grace)
        out[p["product"]] = ("нет, отменяется" if cancels else "нет, не распространяется" if "ev_cash" in exc
                             else "только в пределах лимита" if "ev_limit_exceeded" in exc
                             else "да" if "ev_cash" in scope else "нет данных")
    return out


def expected_q5() -> dict[str, str]:
    out = {}
    for p in yaml_products("installment"):
        terms = {n["id"]: n.get("term") for n in yaml_nodes(p)}
        rate = [e for e in p["events"] if e["rel"] == "TRIGGERS" and terms.get(e["target"]) == "term_rate"]
        rate.sort(key=lambda e: e["event"] != "ev_missed_payment")  # сначала событие из вопроса
        out[p["product"]] = (f"да, после «{EVENT_NAMES[rate[0]['event']].lower()}»" if rate
                             else "нет процентов, другие последствия")
    return out


def expected_q9() -> dict[str, str]:
    out = {}
    for p in yaml_products("student_card"):
        values = {n["id"]: n.get("value", "") for n in yaml_nodes(p)}
        target = next(e["target"] for e in p["events"] if e["event"] == "ev_student_lost" and e["rel"] == "TRIGGERS")
        amounts = [int(re.sub(r"\s", "", m)) for m in re.findall(r"(\d[\d\s]*)\s*₽\s*в месяц", values[target])]
        out[p["product"]] = (f"да, {max(amounts)} ₽ в месяц, если условия не выполнены" if any(amounts)
                             else "нет, обслуживание 0 ₽")
    return out


class VerdictsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        db = Path(cls.tmp.name) / "graph.kuzu"
        build_graph.build(build_graph.ROOT / "data" / "extracted" / "gold", db)
        cls.rec = EntityRecognizer()
        cls.vb = VerdictBuilder(GraphRetriever(db, cls.rec))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def table(self, q):
        return self.vb.build(self.rec.recognize(q))

    def assert_matches_yaml(self, q, expected):
        t = self.table(q)
        self.assertEqual({r.product: r.verdict for r in t.rows}, expected)
        for r in t.rows:  # у каждой строки с выводом есть пункт и дата редакции
            if r.verdict != "нет данных":
                self.assertRegex(r.cite(), r"\.md, п\. \d+\.\d+, ред\. 2026-\d\d-\d\d")

    def test_q4_matches_yaml(self):
        expected = expected_q4()
        self.assertEqual(expected["100 дней"], "да")
        self.assertEqual(expected["Опорная 55"], "только в пределах лимита")
        self.assert_matches_yaml(Q4, expected)

    def test_q4_detail_has_commission_and_limit(self):
        rows = {r.product: r for r in self.table(Q4).rows}
        self.assertIn("3,9% + 390 ₽", rows["100 дней"].detail)
        self.assertIn("50 000 ₽", rows["Опорная 55"].detail)

    def test_q5_matches_yaml(self):
        self.assert_matches_yaml(Q5, expected_q5())
        self.assertIn("Доплата", {r.product: r for r in self.table(Q5).rows}["Кедр Рассрочка"].detail)

    def test_q9_matches_yaml(self):
        self.assert_matches_yaml(Q9, expected_q9())

    def test_no_explicit_fact_means_no_data(self):
        # Без scope у «100 дней» вывод был бы «нет данных», а не «да»
        g = self.vb.g
        orig = g.store.condition_scopes
        g.store.condition_scopes = lambda pid: {}
        try:
            self.assertEqual({r.product: r.verdict for r in self.table(Q4).rows}["100 дней"], "нет данных")
        finally:
            g.store.condition_scopes = orig

    def test_named_product_is_not_a_list_question(self):
        self.assertIsNone(self.table("Что отменяет льготный период по карте «Полярная 120»?"))

    def test_list_prompt_has_only_the_table(self):
        content = build_list_messages(Q4, self.table(Q4))[-1]["content"]
        self.assertNotIn("Кэшбэк", content)
        self.assertNotIn("[c1]", content)

    def test_summary_contradicting_table_is_caught(self):
        # резюме модели с итогового прогона (review/results_final_llm.json, коммит e78fec6)
        q4 = "100 дней подходит для снятия наличных с льготным периодом, другие карты не подходят. Оговорка: у 100 дней есть комиссия."
        self.assertEqual(summary_violations(q4, self.table(Q4)),
                         ["не назван продукт с выводом «только в пределах лимита»: «Опорная 55»"])
        q5 = "Кедр Рассрочка и Рассрочка по кнопке не начисляют проценты, но имеют другие последствия."
        self.assertTrue(any("Рассрочка по кнопке" in v for v in summary_violations(q5, self.table(Q5))))

    def test_code_summary_passes_own_check(self):
        for q in (Q4, Q5, Q9):
            t = self.table(q)
            self.assertEqual(summary_violations(t.summary(), t), [], q)

    def test_summary_violations(self):
        t = self.table(Q4)
        self.assertEqual(summary_violations("«100 дней» сохраняет, но комиссия 3,9% + 390 ₽; «Опорная 55» — до 50 000 ₽.", t), [])
        bad = summary_violations("Сохраняют «100 дней» и «Плати частями», ставка до 59,8%.", t)
        self.assertIn("продукт не из таблицы: плати частями", bad)
        self.assertIn("число не из таблицы: 59,8", bad)
        self.assertIn("не назван продукт с выводом «только в пределах лимита»: «Опорная 55»", bad)


class ListAnswerTest(unittest.TestCase):
    def test_bad_summary_is_replaced_by_code_summary(self):
        def chat(messages, schema=None, **kw):
            self.assertIsNone(schema)  # маршрутизатор не вызывается
            self.assertEqual(kw["max_tokens"], 160)
            return "Льготный период сохраняют «100 дней» и «Меридиан 150», ставка 59,8%."
        agent = Agent(router=Router(chat=chat), chat=chat)
        resp = agent.ask(Q4)
        self.assertEqual(resp.route, "list")
        self.assertNotIn("59,8", resp.answer)
        self.assertIn("Только в пределах лимита: «Опорная 55»", resp.answer)
        self.assertTrue(any("отброшено" in n for n in resp.notes))
        self.assertEqual({s.file for s in resp.sources},
                         {"balt_credit.md", "kedr_credit.md", "merid_credit.md", "opora_credit.md", "sever_credit.md"})
        self.assertTrue(resp.render().startswith("| Продукт |"))
        self.assertIn("digraph", resp.dot)

    def test_without_graph_list_mode_is_off(self):
        calls = []

        def chat(messages, schema=None, **kw):
            calls.append(schema is not None)
            return json.dumps({"route": "rag", "tool": "none", "product_hints": []}) if schema else "Ответ.\nИСПОЛЬЗОВАНО: c1"
        resp = Agent(router=Router(chat=chat), chat=chat).ask(Q4, use_graph=False)
        self.assertEqual(resp.route, "rag")
        self.assertIsNone(resp.table)


if __name__ == "__main__":
    unittest.main()
