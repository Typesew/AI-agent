"""Этап 1: корпус, чанкинг, схема графа, отсутствие старых банков, числа в кейсах."""

import json
import re
import tempfile
import unittest
from pathlib import Path

from corpus import build_corpus
from indexing import build_graph
from indexing.chunk import all_docs, chunk_doc, parse_doc

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".venv", "__pycache__", "chroma", ".git"}
NUM_RE = re.compile(r"(\d[\d\s]*(?:,\d+)?)\s*(₽|%)")


def repo_text_files():
    for path in ROOT.rglob("*"):
        if path.is_file() and not SKIP_DIRS & set(path.relative_to(ROOT).parts) and "graph.kuzu" not in path.name:
            try:
                yield path, path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue


class CorpusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build_corpus.main()
        cls.docs = {parse_doc(p)[0]["doc_id"]: parse_doc(p) for p in all_docs()}

    def test_document_counts(self):
        types = [meta["doc_type"] for meta, _ in self.docs.values()]
        self.assertEqual(sum(t in ("credit_card", "installment", "debit_card", "student_card") for t in types), 20)
        self.assertEqual(types.count("regulatory"), 6)
        self.assertEqual(types.count("case"), 8)
        self.assertEqual(types.count("glossary"), 1)

    def test_five_banks_four_products(self):
        catalog = json.loads(build_corpus.CATALOG.read_text(encoding="utf-8"))
        per_bank = {}
        for p in catalog["products"]:
            per_bank.setdefault(p["bank_id"], set()).add(p["doc_type"])
        self.assertEqual(len(per_bank), 5)
        for types in per_bank.values():
            self.assertEqual(types, {"credit_card", "installment", "debit_card", "student_card"})
        self.assertEqual(len(catalog["terms"]), 34)
        self.assertEqual(len(catalog["events"]), 10)

    def test_disclaimer_everywhere(self):
        for doc_id, (_, body) in self.docs.items():
            self.assertRegex(body, r"не оферта", doc_id)

    def test_chunker_keeps_clauses_whole(self):
        for path in all_docs():
            _, body = parse_doc(path)
            clauses = [l.strip() for l in body.splitlines() if re.match(r"^\d+\.\d+\.", l.strip())]
            chunks = [c["text"] for c in chunk_doc(path)]
            for clause in clauses:
                self.assertEqual(sum(clause in c for c in chunks), 1, f"{path.name}: {clause[:40]}")

    def test_gold_graph_passes_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            errors = build_graph.build(build_graph.ROOT / "data" / "extracted" / "gold", Path(tmp) / "g.kuzu")
        self.assertEqual(errors, [])

    def test_no_old_banks(self):
        # Шаблон собран из кусков, чтобы этот файл сам не находился поиском.
        old = ["пол" + "ярн", "сир" + "ень", "каш" + "тан", "pol" + "ar", "sir" + "en", "kash" + "tan"]
        pattern = re.compile("|".join(old), re.IGNORECASE)
        # То же правило, что у команды grep в README: строка допустима, если в ней есть «Полярная 120»
        # (продукт банка «Северный Путь») или её сокращение в ответах модели — но не «… 100».
        allowed = re.compile("пол" + r"ярн(?:ая|ой|ую)(?: 1[^0]| 120|[^ 0-9]|$)", re.IGNORECASE)
        hits = []
        for path, text in repo_text_files():
            for line in text.splitlines():
                if pattern.search(line) and not allowed.search(line):
                    hits.append(f"{path.relative_to(ROOT)}: {line.strip()[:80]}")
        self.assertEqual(hits, [])

    def test_extra_numbers_come_from_products(self):
        def numbers(text):
            return {re.sub(r"\s+", " ", m.group(1)).strip() + m.group(2) for m in NUM_RE.finditer(text)}

        reference = set()
        for meta, body in self.docs.values():
            if meta["doc_type"] not in ("regulatory", "case"):
                reference |= numbers(body)
        for path in sorted((ROOT / "corpus" / "extra").glob("*.md")):
            unknown = numbers(path.read_text(encoding="utf-8")) - reference
            self.assertEqual(unknown, set(), path.name)


if __name__ == "__main__":
    unittest.main()
