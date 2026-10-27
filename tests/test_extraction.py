"""Этап 6: обвязка LLM-экстракции и метрика без Ollama (LLM подменён эталоном)."""

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from eval import eval_extraction
from indexing import extract_llm
from indexing.chunk import RAW_DIR

GOLD = extract_llm.GOLD_DIR


def gold_llm(doc_id: str, prompts: list[str], swap_cancels: bool = False):
    """Отвечает на промпт раздела эталонными узлами и связями этого раздела, как ответила бы идеальная модель."""
    gold = json.loads((GOLD / f"{doc_id}.json").read_text(encoding="utf-8"))

    def chat(messages, schema=None, **kw):
        prompt = messages[-1]["content"]
        prompts.append(prompt)
        section = re.search(r"\n(\d+)\. ", prompt.split("Фрагмент:")[1]).group(1)
        here = [n for n in gold["nodes"] if n["section"].split(".")[0] == section]
        ids = {n["id"] for n in here}
        known = set(re.findall(r"(\w+__\w+) \(", prompt))

        def ref(x):  # на узлы прошлых разделов модель ссылается по id из промпта
            prev = f"{doc_id}__{x}"
            return prev if prev in known else x

        edges = []
        for rel, a, b, sec in gold["edges"]:
            if sec.split(".")[0] != section:
                continue
            if swap_cancels and rel in ("CANCELS", "TRIGGERS"):
                rel = "TRIGGERS" if rel == "CANCELS" else "CANCELS"
            edges.append({"type": rel, "from": a if a in ids else ref(a), "to": b if b in ids else ref(b), "section": sec})
        nodes = [{k: n[k] for k in ("id", "label", "name", "value", "section")} for n in here]
        return json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False)
    return chat


class ExtractionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()  # сырые ответы подставной модели не попадают в data/
        patcher = mock.patch.object(extract_llm, "RAW_DIR_LLM", Path(self.tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)
        self.catalog = {n["id"]: n for n in json.loads((GOLD / "glossary.json").read_text(encoding="utf-8"))["nodes"]}
        self.names = {i: n["name"] for i, n in self.catalog.items()}

    def extract(self, doc_id, **kw):
        prompts = []
        with mock.patch.object(extract_llm.llm, "chat", side_effect=gold_llm(doc_id, prompts, **kw)):
            result = extract_llm.extract_doc(RAW_DIR / f"{doc_id}.md", self.catalog, "fake")
        return result, prompts

    def score(self, doc_id, result, product):
        gold = json.loads((GOLD / f"{doc_id}.json").read_text(encoding="utf-8"))
        return eval_extraction.score(eval_extraction._edges(gold, self.names, (doc_id, product)),
                                     eval_extraction._edges(result, self.names, (doc_id, product)))

    def test_perfect_model_gets_full_score_with_cross_section_refs(self):
        result, prompts = self.extract("sever_credit")
        tp, n_pred, n_gold = self.score("sever_credit", result, "Полярная 120")
        self.assertEqual((tp, n_pred), (n_gold, n_gold))
        # TRIGGERS из п. 2.3 ссылается на ставку из раздела 1 — по id, переданному в промпте
        self.assertIn("sever_credit__sever_credit_rate (Процентная ставка)", prompts[1])

    def test_prompt_catalog_is_reduced_by_product_type(self):
        _, prompts = self.extract("sever_credit")
        self.assertNotIn("term_sbp", prompts[0])
        self.assertNotIn("ev_student_lost", prompts[0])
        self.assertIn("ev_limit_exceeded", prompts[0])
        _, student = self.extract("kedr_student")
        self.assertIn("ev_student_lost", student[0])

    def test_broken_chunk_is_skipped_and_copies_dropped(self):
        answers = iter(['{"nodes": [{"id": "x', '{"nodes": [{"id": "term_grace", "label": "Condition", "name": "Льготный период", '
                        '"value": "", "section": "2.1"}, {"id": "g", "label": "Condition", "name": "Льготный период", '
                        '"value": "до 100 дней", "section": "2.1"}, {"id": "g2", "label": "Condition", "name": "Льготный период", '
                        '"value": "до 100 дней", "section": "2.1"}], "edges": []}'])
        calls = []
        def chat(messages, **kw):
            calls.append(kw)
            return next(answers, '{"nodes": [], "edges": []}')
        with mock.patch.object(extract_llm.llm, "chat", side_effect=chat):
            result = extract_llm.extract_doc(RAW_DIR / "balt_credit.md", self.catalog, "fake")
        self.assertEqual(result["failed_chunks"], ["balt_credit#1"])
        self.assertEqual([n["value"] for n in result["nodes"]], ["до 100 дней"])  # копия термина и дубль отброшены
        self.assertEqual(calls[0]["max_tokens"], extract_llm.MAX_TOKENS)
        self.assertEqual(extract_llm.SCHEMA["properties"]["nodes"]["maxItems"], extract_llm.MAX_NODES)

    def test_model_copies_of_product_and_events_become_references(self):
        data = {"nodes": [
            {"id": "product_balt_credit", "label": "Condition", "name": "Продукт", "value": "100 дней", "section": "1"},
            {"id": "event_ev_cash", "label": "Condition", "name": "Снятие наличных", "value": "x", "section": "3"},
            {"id": "defined_as", "label": "Condition", "name": "DEFINED_AS", "value": "a -> b", "section": "3"},
            {"id": "fee_3_1", "label": "Fee", "name": "Комиссия", "value": "3,9% + 390 ₽", "section": "3.1"}],
            "edges": [{"type": "CHARGES", "from": "product_balt_credit", "to": "fee_3_1", "section": "3.1"},
                      {"type": "TRIGGERS", "from": "balt_credit__event_ev_cash", "to": "fee_3_1", "section": "3.1"}]}
        nodes, edges = [], []
        extract_llm._merge(data, "balt_credit", self.catalog, nodes, edges)
        self.assertEqual([n["id"] for n in nodes], ["balt_credit__fee_3_1"])
        self.assertEqual(edges, [["CHARGES", "balt_credit", "balt_credit__fee_3_1", "3.1"],
                                 ["TRIGGERS", "ev_cash", "balt_credit__fee_3_1", "3.1"]])

    def test_metric_catches_swapped_cancels_and_triggers(self):
        result, _ = self.extract("sever_credit", swap_cancels=True)
        tp, n_pred, n_gold = self.score("sever_credit", result, "Полярная 120")
        self.assertLess(tp, n_gold)
        self.assertEqual(n_pred, n_gold)


if __name__ == "__main__":
    unittest.main()
