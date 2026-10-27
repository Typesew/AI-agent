"""Этап C: проверки run_eval — форма слова, запреты, части результата, кэш маршрутизатора."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from eval.run_eval import RouterCache, check, contains, forbidden_hits


class ChecksTest(unittest.TestCase):
    def test_contains_ignores_word_form_and_spaces_in_numbers(self):
        self.assertTrue(contains("банк вправе потребовать досрочное погашение", "досрочного погашения"))
        self.assertTrue(contains("отметка в кредитную историю", "кредитной истории"))
        self.assertTrue(contains("переплата 19 817,84 ₽", "19817,84"))
        self.assertFalse(contains("ставка 29,9%", "59,8"))

    def test_forbidden_pair_respects_negation(self):
        bad = "Льготный период сохраняют «100 дней» и «Меридиан 150»."
        ok = "«Меридиан 150», «Кедр 78» и «Полярная 120» льготный период не сохраняют."
        rule = [["Меридиан 150", "сохраня"]]
        self.assertEqual(forbidden_hits(bad, rule), ["Меридиан 150 + сохраня"])
        self.assertEqual(forbidden_hits(ok, rule), [])
        self.assertEqual(forbidden_hits("У рассрочек ставка до 59,8% годовых.", ["59,8%"]), ["59,8%"])

    def test_parts_are_reported_separately(self):
        q = {"id": 4, "type": "cross", "table": {"100 дней": "да"}, "facts": ["100 дней"],
             "forbidden": [["Меридиан 150", "сохраня"]], "sources": ["balt_credit.md"]}
        table = SimpleNamespace(rows=[SimpleNamespace(product="100 дней", verdict="да")],
                                markdown=lambda: "| 100 дней | да |")
        resp = SimpleNamespace(answer="Сохраняют «100 дней» и «Меридиан 150».", route="list", tool_calls=[],
                               sources=[SimpleNamespace(file="balt_credit.md")], table=table, notes=[],
                               render=lambda: "")
        r = check(q, resp)
        self.assertEqual(r["parts"], {"route": True, "table": True, "text": False})
        self.assertFalse(r["passed"])

    def test_router_cache_skips_model_on_repeat(self):
        calls = []

        def chat(messages, schema=None, **kw):
            calls.append(bool(schema))
            return '{"route": "rag"}' if schema else "ответ"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.json"
            for _ in range(2):
                cached = RouterCache(chat, path)
                cached([{"role": "user", "content": "q"}], schema={"type": "object"})
                cached([{"role": "user", "content": "q"}])  # ответ модели не кэшируется
            self.assertEqual(calls, [True, False, False])
            self.assertEqual(len(json.loads(path.read_text(encoding="utf-8"))), 1)


if __name__ == "__main__":
    unittest.main()
