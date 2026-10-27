"""Этап 3: формулы инструментов, граничные случаи, санитизация, MCP stdio."""

import os
import tempfile
import unittest
from pathlib import Path

from mcp_server import tools
from mcp_server.tools import ToolError


def reference_annuity(s, annual_rate, n):
    """Независимая проверка: остаток после n платежей P должен быть нулевым."""
    r = annual_rate / 1200
    lo, hi = 0.0, s
    for _ in range(200):  # бисекция по платежу
        p = (lo + hi) / 2
        balance = s
        for _ in range(n):
            balance = balance * (1 + r) - p
        lo, hi = (p, hi) if balance > 0 else (lo, p)
    return p


class LoanTest(unittest.TestCase):
    def test_annuity_matches_independent_formula(self):
        r = tools.loan_calculator(100_000, 18, 24)
        self.assertAlmostEqual(r["monthly_payment"], reference_annuity(100_000, 18, 24), places=2)
        self.assertAlmostEqual(r["overpayment"], 19_817.84, places=2)  # вопрос 11: ≈ 19 800 ₽

    def test_diff_overpayment(self):
        r = tools.loan_calculator(100_000, 18, 24, "diff")
        s, rate, n = 100_000, 0.18 / 12, 24
        self.assertAlmostEqual(r["overpayment"], s * rate * (n + 1) / 2, places=2)  # 18 750 ₽
        self.assertAlmostEqual(sum(r["schedule"]), r["total"], delta=24 * 0.005)  # округление каждого платежа
        self.assertGreater(r["first_payment"], r["last_payment"])

    def test_zero_rate_installment(self):
        r = tools.loan_calculator(60_000, 0, 12)
        self.assertEqual(r["monthly_payment"], 5_000)  # вопрос 10
        self.assertEqual(r["overpayment"], 0)

    def test_fee_is_added_to_overpayment(self):
        self.assertEqual(tools.loan_calculator(12_000, 0, 12, fee=500)["overpayment"], 500)

    def test_validation(self):
        bad = [(-1, 18, 12, "annuity"), (1000, 150, 12, "annuity"), (1000, 18, 0, "annuity"),
               (1000, 18, 12.5, "annuity"), (1000, 18, 12, "bullet"), ("1000", 18, 12, "annuity"),
               (True, 18, 12, "annuity")]
        for args in bad:
            with self.subTest(args=args), self.assertRaises(ToolError):
                tools.loan_calculator(*args)
        self.assertEqual(tools.loan_calculator(1000, 0, 12.0)["months"], 12)


class GraceTest(unittest.TestCase):
    def test_from_month_start_sever_credit(self):
        r = tools.grace_period_calc("2026-11-15", grace_days=120, mode="from_month_start")
        self.assertEqual(r["start"], "2026-11-01")
        self.assertEqual(r["last_day"], "2027-02-28")  # вопрос 12

    def test_period_is_inclusive(self):
        self.assertEqual(tools.grace_period_calc("2026-03-10", grace_days=1)["last_day"], "2026-03-10")
        self.assertEqual(tools.grace_period_calc("2026-03-10", grace_days=55)["last_day"], "2026-05-03")

    def test_base_plus_extension_merid(self):
        r = tools.grace_period_calc("15.11.2026", mode="base_plus_extension")
        self.assertEqual((r["grace_days"], r["base_days"], r["extension_days"]), (150, 60, 90))  # вопрос 13
        self.assertEqual(r["base_last_day"], "2027-01-13")
        self.assertIn("1,9%", r["extension_cost"])

    def test_statement_day_clamped_to_month_end(self):
        self.assertEqual(tools.grace_period_calc("2026-02-10", 31, 55)["next_statement"], "2026-02-28")
        self.assertEqual(tools.grace_period_calc("2026-02-10", 5, 55)["next_statement"], "2026-03-05")

    def test_validation(self):
        for kwargs in [{"purchase_date": "15/11/2026", "grace_days": 10}, {"purchase_date": "2026-02-30", "grace_days": 10},
                       {"purchase_date": "2026-01-01", "grace_days": 0}, {"purchase_date": "2026-01-01", "grace_days": 400},
                       {"purchase_date": "2026-01-01", "grace_days": 10, "mode": "weekly"},
                       {"purchase_date": "2026-01-01", "grace_days": 10, "statement_day": 32}]:
            with self.subTest(**kwargs), self.assertRaises(ToolError):
                tools.grace_period_calc(**kwargs)


class SummaryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = tools.SUMMARY_DIR
        tools.SUMMARY_DIR = Path(self.tmp.name) / "summaries"

    def tearDown(self):
        tools.SUMMARY_DIR = self.old
        self.tmp.cleanup()

    def test_path_traversal_is_impossible(self):
        for title in ["../../etc/passwd", "/abs/path", "..", "a/../../b", "C:\\Windows\\x", "\x00name"]:
            with self.subTest(title=title):
                path = Path(tools.save_summary(title, "текст")["path"]).resolve()
                self.assertEqual(path.parent, tools.SUMMARY_DIR.resolve())
        self.assertFalse((Path(self.tmp.name) / "etc").exists())

    def test_no_overwrite_and_content(self):
        a = tools.save_summary("Разбор «Полярная 120»", "первый")["path"]
        b = tools.save_summary("Разбор «Полярная 120»", "второй")["path"]
        self.assertNotEqual(a, b)
        self.assertIn("первый", Path(a).read_text(encoding="utf-8"))

    def test_validation(self):
        for args in [("", "x"), ("t", ""), ("t", "x" * 100_001), (None, "x")]:
            with self.assertRaises(ToolError):
                tools.save_summary(*args)


class CompareTest(unittest.TestCase):
    def test_compare_debit_cards(self):
        r = tools.compare_products(["sever_debit", "opora_debit", "balt_debit"])  # вопрос 15
        self.assertIn("99 ₽ в месяц", r["table"])
        self.assertIn("149 ₽ в месяц", r["table"])
        self.assertEqual(r["table"].count("Обслуживание карты |"), 1)
        self.assertTrue(r["sources"])

    def test_unknown_ids(self):
        with self.assertRaises(ToolError):
            tools.compare_products(["sever_debit", "nope"])
        with self.assertRaises(ToolError):
            tools.compare_products(["sever_debit"])


class MCPStdioTest(unittest.TestCase):
    """Сервер стартует подпроцессом; заодно проверяем, что граф читается двумя процессами."""

    def test_client_calls_tools(self):
        from agent.client import MCPClient, ToolCallError
        from rag.graph_retriever import GraphRetriever

        GraphRetriever()  # граф уже открыт в этом процессе на чтение
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["BANK_SUMMARY_DIR"] = tmp
            try:
                with MCPClient() as c:
                    self.assertEqual(set(c.list_tools()),
                                     {"loan_calculator", "grace_period_calc", "compare_products", "save_summary"})
                    self.assertEqual(c.call("loan_calculator", {"sum": 60000, "rate": 0, "months": 12})["monthly_payment"], 5000)
                    self.assertIn("table", c.call("compare_products", {"ids": ["opora_credit", "merid_credit"]}))
                    self.assertTrue(c.call("save_summary", {"title": "тест", "text": "ok"})["path"].endswith(".md"))
                    with self.assertRaises(ToolCallError) as err:
                        c.call("grace_period_calc", {"purchase_date": "вчера", "grace_days": 10})
                    self.assertIn("ГГГГ-ММ-ДД", str(err.exception))
                self.assertEqual(len(list(Path(tmp).glob("*.md"))), 1)
            finally:
                del os.environ["BANK_SUMMARY_DIR"]


if __name__ == "__main__":
    unittest.main()
