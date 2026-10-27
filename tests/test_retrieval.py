"""Этап 2: векторный RAG и BM25 без LLM.

Без Ollama индекс строится хеш-эмбеддером (лексическая заглушка): тест проверяет
обвязку Chroma, фильтры по метаданным и RRF, а не качество bge-m3.
"""

import tempfile
import unittest
from pathlib import Path

from indexing import embed_index
from rag.entities import EntityRecognizer
from rag.retriever import HybridRetriever
from rag.vector_retriever import BM25, VectorRetriever

# (вопрос, документ (или любой из кортежа) в top-5, раздел или None)
CASES = [
    ("Что такое льготный период и когда он пропадает?", ("reg_grace.md", "glossary.md"), None),
    ("Что отменяет льготный период по карте «Полярная 120»?", "sever_credit.md", "2. Льготный период"),
    ("Какие операции не участвуют в кэшбэке по «Полярная 120»?", "sever_credit.md", "4. Кэшбэк и бонусы"),
    ("Когда у «Опорная Black» платное обслуживание?", "opora_debit.md", "1. Обслуживание"),
    ("Что будет с «Студенческая Кедр», когда мне исполнится 25?", "kedr_student.md", "4. Статус и возраст"),
    ("Объясни, что будет, если пропущу платёж по «Плати частями», и что при двух пропусках.", "sever_install.md", "3. Просрочка"),
    ("Что такое ПСК и где её смотреть?", "reg_psk.md", "2. Где её смотреть"),
    ("Чем «Кедр Рассрочка» отличается от обычной рассрочки?", "kedr_install.md", "1. Общие условия"),
]


class RetrievalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        path = Path(cls.tmp.name) / "chroma"
        embed_index.build("hash", path)
        cls.vector = VectorRetriever(path)
        cls.bm25_only = VectorRetriever(Path(cls.tmp.name) / "missing")
        cls.rec = EntityRecognizer()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check(self, retriever):
        for q, doc, section in CASES:
            hits = retriever.search(q, self.rec.recognize(q))
            found = {(c.metadata["source_file"], c.metadata["section"]) for c in hits}
            with self.subTest(q=q):
                docs = doc if isinstance(doc, tuple) else (doc,)
                self.assertTrue(set(docs) & {f for f, _ in found}, found)
                if section:
                    self.assertIn((doc, section), found)

    def test_hybrid_dense_and_bm25(self):
        self.assertEqual(self.vector.mode, "bm25+dense(hash)")
        self.check(self.vector)

    def test_bm25_only_fallback(self):
        self.assertEqual(self.bm25_only.mode, "bm25")
        self.assertIn("не построен", self.bm25_only.note)
        self.check(self.bm25_only)

    def test_metadata_filter_by_product(self):
        q = "Какая комиссия за снятие наличных по «Кедр 78»?"
        files = {c.metadata["source_file"] for c in self.vector.search(q, self.rec.recognize(q))}
        self.assertTrue(files <= {"kedr_credit.md", "glossary.md"} | {f for f in files if f.startswith(("reg_", "case_"))})
        self.assertIn("kedr_credit.md", files)

    def test_chunk_metadata(self):
        meta = self.vector.search("льготный период", None)[0].metadata
        for key in ("bank", "product", "doc_type", "section", "source_file", "date", "page"):
            self.assertIn(key, meta)

    def test_hybrid_adds_graph_facts(self):
        h = HybridRetriever(vector=self.vector, recognizer=self.rec)
        ctx = h.retrieve("Что отменяет льготный период по карте «Полярная 120»?")
        self.assertTrue(ctx.graph_used)
        self.assertTrue(any(f.kind == "cancels" for f in ctx.facts))
        no_graph = h.retrieve("Что отменяет льготный период по карте «Полярная 120»?", use_graph=False)
        self.assertEqual(no_graph.facts, [])

    def test_bm25_math(self):
        bm = BM25([["a", "b"], ["b", "c", "c"]])
        s = bm.scores(["c"])
        self.assertEqual(s[0], 0.0)
        self.assertGreater(s[1], 0.0)


if __name__ == "__main__":
    unittest.main()
