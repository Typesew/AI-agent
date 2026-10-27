"""Единственный модуль, который знает про Kuzu и Cypher.

Kuzu архивирован, версия закреплена (kuzu==0.11.3). Чтобы перенести граф в
Neo4j / FalkorDB / Memgraph, достаточно переписать этот файл: остальной код
работает через методы GraphStore и получает списки строк.
"""

import shutil
from pathlib import Path

import kuzu

from indexing.schema import ddl

# --- запросы на чтение ---------------------------------------------------

Q_CATALOG = "MATCH (n:{label}) RETURN n.id, n.name, n.value, n.aliases"
Q_PRODUCT_TYPES = "MATCH (p:Product)-[:DEFINED_AS]->(t:Term) RETURN p.id, t.id"
Q_CONDITIONS = """
MATCH (p:Product {id: $pid})-[r:HAS_CONDITION]->(c:Condition)
OPTIONAL MATCH (c)-[:DEFINED_AS]->(t:Term)
RETURN c.id, c.name, c.value, t.id"""
Q_FEES = """
MATCH (p:Product {id: $pid})-[r:CHARGES]->(f:Fee)
OPTIONAL MATCH (f)-[:DEFINED_AS]->(t:Term)
RETURN f.id, f.name, f.value, t.id"""
Q_EXCEPTIONS = """
MATCH (p:Product {id: $pid})-[:HAS_CONDITION]->(c:Condition)-[:HAS_EXCEPTION]->(x:Exception)
OPTIONAL MATCH (c)-[:DEFINED_AS]->(t:Term)
OPTIONAL MATCH (x)-[:APPLIES_TO]->(e:Event)
RETURN x.id, c.name, x.name, t.id, e.id"""
Q_CANCELS = """
MATCH (p:Product {id: $pid})-[:HAS_CONDITION]->(c:Condition)<-[r:CANCELS]-(e:Event)
OPTIONAL MATCH (c)-[:DEFINED_AS]->(t:Term)
RETURN c.id, e.id, e.name, c.name, t.id, r.section"""
Q_TRIGGERS = """
MATCH (p:Product {id: $pid})-[:HAS_CONDITION|:CHARGES]->(n)<-[r:TRIGGERS]-(e:Event)
OPTIONAL MATCH (n)-[:DEFINED_AS]->(t:Term)
RETURN n.id, e.id, e.name, n.name, n.value, t.id, r.section"""
Q_SOURCE = "MATCH (n)-[s:SOURCE]->(d:Document) WHERE n.id = $id RETURN d.file, s.section, d.date"
Q_STATS = "MATCH (n) RETURN label(n), count(*)"

# --- запросы на запись ---------------------------------------------------

W_DOCUMENT = "CREATE (:Document {id: $id, title: $title, date: $date, file: $file})"
W_NODE = "CREATE (:{label} {{id: $id, name: $name, value: $value, aliases: $aliases}})"
W_EDGE = "MATCH (a:{la} {{id: $a}}), (b:{lb} {{id: $b}}) CREATE (a)-[:{rel} {{section: $s}}]->(b)"
W_SOURCE = "MATCH (n:{label} {{id: $n}}), (d:Document {{id: $d}}) CREATE (n)-[:SOURCE {{section: $s}}]->(d)"


class GraphStore:
    def __init__(self, db_path: Path, read_only: bool = True):
        self.db = kuzu.Database(str(db_path), read_only=read_only)
        self.conn = kuzu.Connection(self.db)

    @classmethod
    def create(cls, db_path: Path) -> "GraphStore":
        """Новая пустая БД со схемой (старая удаляется)."""
        for p in (db_path, Path(f"{db_path}.wal")):
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        store = cls(db_path, read_only=False)
        for stmt in ddl():
            store.conn.execute(stmt)
        return store

    def rows(self, query: str, params: dict | None = None) -> list[list]:
        result = self.conn.execute(query, params or {})
        out = []
        while result.has_next():
            out.append(result.get_next())
        return out

    # запись
    def add_document(self, doc: dict):
        self.conn.execute(W_DOCUMENT, doc)

    def add_node(self, label: str, node: dict):
        self.conn.execute(W_NODE.format(label=label), {k: node.get(k, "") for k in ("id", "name", "value", "aliases")})

    def add_edge(self, rel: str, la: str, a: str, lb: str, b: str, section: str):
        self.conn.execute(W_EDGE.format(rel=rel, la=la, lb=lb), {"a": a, "b": b, "s": section})

    def add_source(self, label: str, node_id: str, doc_id: str, section: str):
        self.conn.execute(W_SOURCE.format(label=label), {"n": node_id, "d": doc_id, "s": section})

    # чтение
    def catalog(self, label: str) -> list[list]:
        return self.rows(Q_CATALOG.format(label=label))

    def product_types(self) -> dict[str, str]:
        return dict(self.rows(Q_PRODUCT_TYPES))

    def conditions(self, pid: str):
        return self.rows(Q_CONDITIONS, {"pid": pid})

    def fees(self, pid: str):
        return self.rows(Q_FEES, {"pid": pid})

    def exceptions(self, pid: str):
        return self.rows(Q_EXCEPTIONS, {"pid": pid})

    def cancels(self, pid: str):
        return self.rows(Q_CANCELS, {"pid": pid})

    def triggers(self, pid: str):
        return self.rows(Q_TRIGGERS, {"pid": pid})

    def sources(self, node_id: str):
        return self.rows(Q_SOURCE, {"id": node_id})

    def stats(self) -> dict[str, int]:
        return dict(self.rows(Q_STATS))
