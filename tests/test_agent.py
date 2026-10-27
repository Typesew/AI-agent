"""Этап 4: разбор ответа, проверки маршрутизатора, сквозной прогон агента без Ollama.

LLM подменяется: маршрут — эталон из eval/questions.json, «ответ» — весь контекст.
Так проверяются распознавание, ретриверы, граф, MCP, числа и источники; качество
формулировок Qwen3 этим тестом не проверяется (для этого python -m eval.run_eval --llm).
"""

import json
import os
import tempfile
import unittest
from datetime import date

from agent.agent import Agent
from agent.router import Router
from eval.run_eval import QUESTIONS, OfflineLLM, check
from rag.entities import EntityRecognizer
from rag.prompt import parse_answer


class ParseAnswerTest(unittest.TestCase):
    items = {"c1": ("", None), "c2": ("", None), "f1": ("", None)}

    def test_used_line_is_cut_and_parsed(self):
        body, used, ok = parse_answer("**Итог**\nТекст.\nИСПОЛЬЗОВАНО: c2, F1", self.items)
        self.assertEqual((body, used, ok), ("**Итог**\nТекст.", ["c2", "f1"], True))

    def test_missing_or_invalid_ids_mean_all_sources(self):
        self.assertEqual(parse_answer("Текст без строки", self.items)[1:], (list(self.items), False))
        body, used, ok = parse_answer("Текст\nИСПОЛЬЗОВАНО: c9, x1", self.items)
        self.assertEqual((body, used, ok), ("Текст", list(self.items), False))

    def test_think_block_removed(self):
        body, _, _ = parse_answer("<think>рассуждаю</think>Ответ\nИСПОЛЬЗОВАНО: [c1]", self.items)
        self.assertEqual(body, "Ответ")


class RouterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rec = EntityRecognizer()

    def route(self, question, llm_answer):
        router = Router(chat=lambda *a, **k: json.dumps({"product_hints": [], **llm_answer}), today=date(2026, 10, 25))
        return router.route(question, self.rec.recognize(question))

    def test_invented_numbers_ask_for_clarification(self):
        r = self.route("Посчитай переплату по кредиту под 18% на 2 года",
                       {"route": "tool", "tool": "loan_calculator", "sum": 300000, "rate": 18, "months": 24})
        self.assertIsNotNone(r.clarify)

    def test_years_to_months_accepted(self):
        r = self.route("Кредит 100 000 ₽ на 2 года под 18%",
                       {"route": "tool", "tool": "loan_calculator", "sum": 100000, "rate": 18, "months": 24})
        self.assertIsNone(r.clarify)
        self.assertEqual(r.args, {"sum": 100000.0, "rate": 18.0, "months": 24, "type": "annuity"})

    def test_installment_rate_defaults_to_zero(self):
        r = self.route("60 000 ₽ на 12 месяцев в «Меридиан Частями»",
                       {"route": "tool", "tool": "loan_calculator", "sum": 60000, "rate": None, "months": 12})
        self.assertEqual((r.route, r.args["rate"]), ("both", 0.0))

    def test_grace_params_come_from_catalog_not_llm(self):
        r = self.route("Купил 15 ноября 2026 по «Полярная 120»",
                       {"route": "tool", "tool": "grace_period_calc", "purchase_date": "2026-11-15"})
        self.assertEqual(r.args, {"purchase_date": "2026-11-15", "grace_days": 120, "mode": "from_month_start"})

    def test_grace_without_card_asks(self):
        self.assertIsNotNone(self.route("До какой даты вернуть без процентов?",
                                        {"route": "tool", "tool": "grace_period_calc", "purchase_date": "2026-11-15"}).clarify)

    def test_refuse_with_domain_entities_falls_back_to_rag(self):
        self.assertEqual(self.route("Что такое льготный период?", {"route": "refuse", "tool": "none"}).route, "rag")
        self.assertEqual(self.route("Какая погода завтра?", {"route": "refuse", "tool": "none"}).route, "refuse")

    def test_broken_llm_output_falls_back_to_rag(self):
        router = Router(chat=lambda *a, **k: "это не JSON")
        self.assertEqual(router.route("вопрос", self.rec.recognize("вопрос")).route, "rag")

    def test_code_forces_compare_when_llm_chose_rag(self):
        r = self.route("Сравни обслуживание «Северная Карта» и «Яркая Карта»", {"route": "rag", "tool": "none"})
        self.assertEqual((r.route, r.tool, r.args), ("both", "compare_products", {"ids": ["sever_debit", "balt_debit"]}))

    def test_code_forces_grace_calc_when_llm_chose_rag(self):
        r = self.route("Сколько дней грейса у «Меридиан 150» с продлением?", {"route": "rag", "tool": "none"})
        self.assertEqual((r.tool, r.args["mode"], r.args["grace_days"]), ("grace_period_calc", "base_plus_extension", 150))
        self.assertIn("распознан кодом", r.note)
        # на прогоне Qwen3 ответила refuse: отказ понижается до rag, и правило всё равно срабатывает
        refused = self.route("Сколько дней грейса у «Меридиан 150» с продлением?", {"route": "refuse", "tool": "none"})
        self.assertEqual(refused.tool, "grace_period_calc")
        plain = self.route("Что отменяет льготный период по карте «Полярная 120»?", {"route": "rag", "tool": "none"})
        self.assertIsNone(plain.tool)

    def test_compare_needs_two_products(self):
        r = self.route("Сравни «Кедр 78»", {"route": "rag", "tool": "compare_products"})
        self.assertIsNotNone(r.clarify)


class AgentEndToEndTest(unittest.TestCase):
    """12 вопросов сквозняком: агент + MCP-подпроцесс + Chroma/BM25 + граф; LLM подменён."""

    IDS = [2, 4, 5, 6, 9, 10, 11, 12, 14, 18, 19, 20]

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        os.environ["BANK_SUMMARY_DIR"] = cls.tmp.name
        cls.llm = OfflineLLM()
        cls.agent = Agent(router=Router(chat=cls.llm), chat=cls.llm)

    @classmethod
    def tearDownClass(cls):
        cls.agent.close()
        del os.environ["BANK_SUMMARY_DIR"]
        cls.tmp.cleanup()

    def test_questions(self):
        questions = {q["id"]: q for q in json.loads(QUESTIONS.read_text(encoding="utf-8"))}
        for qid in self.IDS:
            q = questions[qid]
            self.llm.router_answer = q["router"]
            with self.subTest(id=qid, q=q["question"]):
                result = check(q, self.agent.ask(q["question"]))
                self.assertTrue(result["passed"], result["problems"])

    def test_sources_block_rendered_once(self):
        self.llm.router_answer = {"route": "rag", "tool": "none"}
        text = self.agent.ask("Что отменяет льготный период по карте «Полярная 120»?").render()
        self.assertEqual(text.count("**Источники**"), 1)
        self.assertIn("ред. 2026-09-01", text)
        self.assertNotIn("ИСПОЛЬЗОВАНО", text)


if __name__ == "__main__":
    unittest.main()
