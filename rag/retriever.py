"""Гибридный ретривер: фрагменты документов (Chroma + BM25) и факты графа.

Граф — второй источник: если его нет или он выключен флагом, ретривер
работает только на векторном поиске, и продукт остаётся рабочим.
"""

from dataclasses import dataclass, field
from pathlib import Path

from rag.entities import Entities, EntityRecognizer
from rag.graph_retriever import DEFAULT_DB, Fact, GraphRetriever, to_dot
from rag.vector_retriever import Chunk, VectorRetriever


@dataclass
class Context:
    entities: Entities
    chunks: list[Chunk] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)
    table: str = ""
    vector_mode: str = ""
    graph_used: bool = False

    @property
    def dot(self) -> str:
        return to_dot(self.facts) if self.facts else ""

    def is_empty(self) -> bool:
        return not (self.chunks or self.facts)


class HybridRetriever:
    def __init__(self, vector: VectorRetriever | None = None, graph: GraphRetriever | None = None,
                 recognizer: EntityRecognizer | None = None, graph_db: Path = DEFAULT_DB):
        self.recognizer = recognizer or EntityRecognizer()
        self.vector = vector or VectorRetriever()
        self.graph = graph
        if self.graph is None and graph_db.exists():
            self.graph = GraphRetriever(graph_db, self.recognizer)

    def retrieve(self, question: str, entities: Entities | None = None, use_graph: bool = True, k: int = 5) -> Context:
        ent = entities or self.recognizer.recognize(question)
        ctx = Context(ent, self.vector.search(question, ent, k=k), vector_mode=self.vector.mode)
        if use_graph and self.graph is not None:
            g = self.graph.retrieve(question, ent)
            ctx.facts, ctx.table, ctx.graph_used = g.facts, g.table, True
        return ctx
