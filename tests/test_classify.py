"""Этап A: извлечение графа классификацией пунктов (LLM подменён)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from eval import eval_extraction
from eval.extraction_ceiling import SAMPLE, ceiling_graph, oracle_answers
from indexing import classify_clauses as cc
from indexing.chunk import RAW_DIR, clauses
from indexing.extract_llm import DOC_TYPE_EVENTS, DOC_TYPE_TERMS, GOLD_DIR

CATALOG = {n["id"]: n for n in json.loads((GOLD_DIR / "glossary.json").read_text(encoding="utf-8"))["nodes"]}


def item(kind="none", term=None, event=None, rel="none", parent=None):
    return {"kind": kind, "term": term, "event": event, "event_relation": rel, "parent_clause": parent}


class ClassifyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def classifier(self, doc_type="credit_card", use_cache=False):
        return cc.ClauseClassifier(CATALOG, DOC_TYPE_TERMS[doc_type], DOC_TYPE_EVENTS[doc_type], "fake",
                                   Path(self.tmp.name), use_cache=use_cache)

    def run_doc(self, doc_id, answer_for, doc_type="credit_card", **kw):
        prompts = []

        def chat(messages, **k):
            prompts.append((messages[-1]["content"], k))
            clause_id = messages[-1]["content"].rsplit("Пункт ", 1)[1].split(" ", 1)[0]
            return answer_for(clause_id)
        with mock.patch.object(cc.llm, "chat", side_effect=chat):
            result, stats = self.classifier(doc_type, **kw).classify_doc(RAW_DIR / f"{doc_id}.md")
        return result, stats, prompts

    def test_ids_names_and_values_are_assigned_by_code(self):
        def answer(cid):
            if cid == "2.1":
                return json.dumps({"items": [item("condition", "term_grace")]})
            if cid == "3.1":
                return json.dumps({"items": [item("fee", "term_cash_withdrawal", "ev_cash", "triggers"),
                                             item("fee", "term_cc_transfer", "ev_transfer", "triggers")]})
            return json.dumps({"items": [item()]})
        result, stats, prompts = self.run_doc("balt_credit", answer)
        ids = {n["id"]: n for n in result["nodes"]}
        self.assertEqual(set(ids), {"balt_credit__2.1", "balt_credit__3.1_1", "balt_credit__3.1_2"})
        self.assertEqual(ids["balt_credit__2.1"]["name"], "Льготный период")  # из справочника
        self.assertIn("до 100 дней", ids["balt_credit__2.1"]["value"])  # текст пункта
        self.assertIn(["TRIGGERS", "ev_cash", "balt_credit__3.1_1", "3.1"], result["edges"])
        self.assertEqual(prompts[0][1]["max_tokens"], cc.MAX_TOKENS)
        # модель не видит эталон: в промпте нет id эталонных узлов
        self.assertNotIn("balt_credit_grace", "".join(p for p, _ in prompts))

    def test_invalid_edges_are_dropped_and_counted(self):
        def answer(cid):
            if cid == "1.3":  # отмена события, нацеленная на платёж: CANCELS -> Fee запрещена схемой
                return json.dumps({"items": [item("fee", "term_service_fee", "ev_promo_end", "cancels")]})
            if cid == "2.2":  # исключение без пункта-условия
                return json.dumps({"items": [item("exception", parent="1.3")]})
            return json.dumps({"items": [item()]})
        result, stats, _ = self.run_doc("balt_credit", answer)
        self.assertFalse(any(e[0] == "CANCELS" for e in result["edges"]))
        self.assertTrue(any("HAS_EXCEPTION" in d for d in stats.dropped_edges) or
                        any("Exception" in d for d in stats.dropped_edges))
        self.assertGreaterEqual(len(stats.dropped_edges), 1)

    def test_values_outside_enums_do_not_crash(self):
        def answer(cid):
            if cid == "1.1":
                return json.dumps({"items": [item("condition", "term_made_up", "ev_nope", "explodes", "9.9"),
                                             "не объект"]})
            if cid == "1.2":
                return '{"items": [{"kind": "cond'  # обрезанный JSON
            return json.dumps({"items": [item()]})
        result, stats, _ = self.run_doc("balt_credit", answer)
        self.assertEqual(stats.invalid_values, 4)
        self.assertEqual(stats.failed, ["1.2"])
        node = next(n for n in result["nodes"] if n["id"] == "balt_credit__1.1")
        self.assertEqual(node["label"], "Condition")

    def test_raw_cache_goes_to_given_dir_and_is_reused(self):
        calls = []

        def answer(cid):
            calls.append(cid)
            return json.dumps({"items": [item()]})
        self.run_doc("kedr_student", answer, doc_type="student_card")
        files = list(Path(self.tmp.name).glob("*.json"))
        self.assertEqual(len(files), len(clauses(RAW_DIR / "kedr_student.md")))
        n = len(calls)
        self.run_doc("kedr_student", answer, doc_type="student_card", use_cache=True)
        self.assertEqual(len(calls), n)  # второй раз модель не звали

    def test_exception_cue_rule(self):
        self.assertTrue(cc.EXCEPTION_CUE.search("Кэшбэк не начисляется на: переводы; снятие наличных."))
        self.assertTrue(cc.EXCEPTION_CUE.search("Снятие наличных и переводы в льготный период не входят."))
        # комиссия с оговоркой про проценты — не перечень исключений
        self.assertFalse(cc.EXCEPTION_CUE.search("Снятие наличных — комиссия 3,9% + 390 ₽; проценты не начисляются."))

    def test_exception_elements(self):
        events = {e: [p for p in CATALOG[e]["aliases"].split("|") if p] for e in ("ev_cash", "ev_transfer")}
        listed = cc.exception_elements("Льготный период не распространяется на: снятие наличных; переводы с карты.", events)
        self.assertEqual([e for _, e in listed], ["ev_cash", "ev_transfer"])
        joined = cc.exception_elements("Снятие наличных и переводы в льготный период не входят.", events)
        self.assertEqual(sorted(e for _, e in joined), ["ev_cash", "ev_transfer"])

    def test_ceiling_of_scheme_on_sample(self):
        """Идеальные ответы (из эталона) через код построения графа: строгий F1 ≥ 0,95."""
        tp = n_pred = n_gold = 0
        for doc in SAMPLE:
            gold = eval_extraction.strict_edges(json.loads((GOLD_DIR / f"{doc}.json").read_text(encoding="utf-8")), doc)
            pred = eval_extraction.strict_edges(ceiling_graph(doc, CATALOG), doc)
            a, b, c = eval_extraction.strict_score(gold, pred)
            tp, n_pred, n_gold = tp + a, n_pred + b, n_gold + c
        p, r = tp / n_pred, tp / n_gold
        self.assertGreaterEqual(2 * p * r / (p + r), 0.95)
        self.assertTrue(all(len(v) <= cc.MAX_ITEMS for v in oracle_answers("balt_credit").values()))


if __name__ == "__main__":
    unittest.main()
