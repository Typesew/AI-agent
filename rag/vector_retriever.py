"""Векторный поиск (Chroma) + BM25 со слиянием через Reciprocal Rank Fusion.

Если индекса Chroma нет или Ollama недоступен для эмбеддинга запроса,
работает только BM25 и это видно в поле mode — без молчаливой подмены.
"""

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from indexing import embed_index
from indexing.chunk import all_docs, chunk_doc
from rag import llm
from rag.entities import Entities
from rag.text import tokens

GENERAL_TYPES = ["glossary", "regulatory", "case"]
TYPE_TO_DOC = {"term_credit_card": "credit_card", "term_installment": "installment",
               "term_debit_card": "debit_card", "term_student_card": "student_card"}
STOPWORDS = {"что", "как", "для", "при", "или", "если", "это", "мне", "все", "его", "так", "чем", "под", "над",
             "был", "без", "где", "когда", "какой", "какие", "какая", "будет", "сколько"}
RRF_K = 60
CANDIDATES = 20


def bm25_terms(text: str) -> list[str]:
    # Для BM25 нужен одинаковый стем у всех форм слова, поэтому фиксированный префикс
    # из 5 букв: «кэшбэк» и «кэшбэке» -> «кэшбэ» (в rag.text длина зависит от слова).
    return [t if t.isdigit() else t[:5] for t in tokens(text) if t not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / max(len(docs), 1)
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, ln in zip(self.tf, self.len):
            s = 0.0
            for t in query:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            out.append(s)
        return out


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict
    score: float = 0.0
    ranks: dict = field(default_factory=dict)  # {"bm25": 1, "dense": 3}


def allowed_filter(ent: Entities | None) -> tuple[set[str] | None, set[str] | None]:
    """(doc_id, doc_type), по которым ограничиваем поиск; None — без ограничения."""
    if ent is None:
        return None, None
    if ent.definition and not ent.products:
        return set(), set(GENERAL_TYPES)
    if ent.products:
        return set(ent.products), set(GENERAL_TYPES)
    if ent.type_terms:
        return set(), {TYPE_TO_DOC[t] for t in ent.type_terms if t in TYPE_TO_DOC} | set(GENERAL_TYPES)
    return None, None


class VectorRetriever:
    def __init__(self, chroma_dir: Path = embed_index.CHROMA_DIR):
        self.chunks = [c for p in all_docs() for c in chunk_doc(p)]
        self.bm25 = BM25([bm25_terms(c["text"]) for c in self.chunks])
        self.collection, self.embed, self.mode = None, None, "bm25"
        self.note = ""
        try:
            col = embed_index.client(chroma_dir).get_collection(embed_index.COLLECTION)
            name = (col.metadata or {}).get("embedder", llm.EMBED_MODEL)
            if name != "hash" and not llm.available():
                self.note = "Ollama недоступен: эмбеддинг запроса невозможен, работает только BM25"
            else:
                self.collection, self.embed = col, embed_index.embedder(name)
                self.mode = f"bm25+dense({name})"
        except Exception:  # индекса нет
            self.note = "индекс Chroma не построен (python -m indexing.embed_index), работает только BM25"

    def _ok(self, meta: dict, ids: set[str] | None, types: set[str] | None) -> bool:
        if ids is None and types is None:
            return True
        return meta["doc_id"] in (ids or set()) or meta["doc_type"] in (types or set())

    def search(self, question: str, entities: Entities | None = None, k: int = 5) -> list[Chunk]:
        ids, types = allowed_filter(entities)
        ranked: dict[str, Chunk] = {}

        scores = self.bm25.scores(bm25_terms(question))
        order = sorted((i for i, c in enumerate(self.chunks) if scores[i] > 0 and self._ok(c["metadata"], ids, types)),
                       key=lambda i: -scores[i])[:CANDIDATES]
        for rank, i in enumerate(order, 1):
            c = self.chunks[i]
            ranked.setdefault(c["id"], Chunk(c["id"], c["text"], c["metadata"])).ranks["bm25"] = rank

        if self.collection is not None:
            where = None
            if ids is not None or types is not None:
                where = {"$or": [{"doc_id": {"$in": sorted(ids or {"-"})}},
                                 {"doc_type": {"$in": sorted(types or {"-"})}}]}
            res = self.collection.query(query_embeddings=self.embed([question]), n_results=CANDIDATES, where=where)
            for rank, (cid, text, meta) in enumerate(zip(res["ids"][0], res["documents"][0], res["metadatas"][0]), 1):
                ranked.setdefault(cid, Chunk(cid, text, meta)).ranks["dense"] = rank

        for c in ranked.values():
            c.score = sum(1 / (RRF_K + r) for r in c.ranks.values())
        return sorted(ranked.values(), key=lambda c: -c.score)[:k]

    def lexical_hit(self, question: str) -> float:
        """Максимальный BM25 по всему корпусу: 0 — в корпусе нет ни одного слова вопроса."""
        return max(self.bm25.scores(bm25_terms(question)), default=0.0)
