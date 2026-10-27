"""Загрузка извлечённых узлов и связей в графовую БД.

    python -m indexing.build_graph                  # эталонная разметка (data/extracted/gold)
    python -m indexing.build_graph --source llm     # результат indexing.extract_llm

Банк, продукт, документ и связи OFFERED_BY / DEFINED_AS(тип продукта) берутся
из front-matter документа детерминированно, их не нужно доверять LLM.
Регуляторика и разборы кейсов (doc_type regulatory / case) в граф не входят,
они есть только в векторном индексе.
"""

import argparse
import json
from pathlib import Path

from indexing.chunk import all_docs, parse_doc
from indexing.schema import check_edge
from rag.graph_store import GraphStore

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "graph.kuzu"
VECTOR_ONLY_TYPES = {"regulatory", "case"}


def skeleton(meta: dict) -> tuple[list[dict], list[list[str]]]:
    if meta.get("doc_type") == "glossary":
        return [], []
    pid, bid = meta["doc_id"], meta["bank_id"]
    nodes = [
        {"id": bid, "label": "Bank", "name": meta["bank"], "value": "", "aliases": "", "section": ""},
        {"id": pid, "label": "Product", "name": meta["product"], "value": meta["doc_type"],
         "aliases": meta.get("aliases", "").replace("-", ""), "section": ""},
    ]
    edges = [["OFFERED_BY", pid, bid]]
    if meta.get("product_term", "-") != "-":
        edges.append(["DEFINED_AS", pid, meta["product_term"]])
    return nodes, edges


def collect(source_dir: Path):
    """Собирает узлы всех документов, проверяет связи по схеме."""
    documents, nodes, occurrences, edges, errors = [], {}, [], [], []
    for path in all_docs():
        meta, _ = parse_doc(path)
        if meta.get("doc_type") in VECTOR_ONLY_TYPES:
            continue
        doc_id = meta["doc_id"]
        extracted_path = source_dir / f"{doc_id}.json"
        if not extracted_path.exists():
            errors.append(f"{doc_id}: нет файла {extracted_path.name}")
            continue
        extracted = json.loads(extracted_path.read_text(encoding="utf-8"))
        title = f"{meta['product']} ({meta['bank']})" if meta["doc_type"] != "glossary" else "Глоссарий"
        documents.append({"id": doc_id, "title": title, "date": meta["date"], "file": meta["source_file"]})

        sk_nodes, sk_edges = skeleton(meta)
        for node in sk_nodes + extracted["nodes"]:
            nodes.setdefault(node["id"], node)  # общий узел (банк, событие) создаём один раз
            occurrences.append((node["id"], doc_id, node.get("section", "")))
        edges += [(doc_id, e) for e in sk_edges + extracted["edges"]]

    labels = {nid: n["label"] for nid, n in nodes.items()}
    valid = []
    for doc_id, edge in edges:
        rel, a, b = edge[:3]
        err = check_edge(rel, labels.get(a), labels.get(b))
        if err:
            errors.append(f"{doc_id}: {rel} {a} -> {b}: {err}")
        else:
            valid.append((rel, a, b, edge[3] if len(edge) > 3 else ""))
    return documents, nodes, occurrences, sorted(set(valid)), errors


def build(source_dir: Path, db_path: Path) -> list[str]:
    documents, nodes, occurrences, edges, errors = collect(source_dir)

    store = GraphStore.create(db_path)
    for d in documents:
        store.add_document(d)
    for n in nodes.values():
        store.add_node(n["label"], n)
    for rel, a, b, section in edges:
        store.add_edge(rel, nodes[a]["label"], a, nodes[b]["label"], b, section)
    for node_id, doc_id, section in set(occurrences):
        store.add_source(nodes[node_id]["label"], node_id, doc_id, section)

    print(f"Граф {db_path.name}: документов {len(documents)}, узлов {len(nodes)}, связей {len(edges)}")
    if errors:
        print(f"Отброшено {len(errors)}:")
        for e in errors:
            print("  -", e)
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["gold", "llm"], default="gold")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()
    build(ROOT / "data" / "extracted" / args.source, args.db)


if __name__ == "__main__":
    main()
