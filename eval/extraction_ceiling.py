"""Потолок режима classify: какой F1 дал бы идеальный классификатор пунктов.

    python -m eval.extraction_ceiling [--docs balt_credit sever_install ...]

Ответы «идеального» классификатора выводятся из эталона (data/extracted/gold)
и проходят через тот же код построения графа, что и ответы модели. Это только
анализ схемы: модель эталон не видит. Разница между потолком и результатом
модели — ошибки модели; разница между потолком и 1,00 — то, что схема
(один пункт → до 3 элементов из перечислений) выразить не может.
"""

import argparse
import json
import tempfile
from pathlib import Path

from eval import eval_extraction
from indexing.chunk import RAW_DIR, all_docs, clauses, parse_doc
from indexing.classify_clauses import MAX_ITEMS, Stats, build_graph
from indexing.extract_llm import DOC_TYPE_EVENTS, GOLD_DIR

SAMPLE = ["balt_credit", "sever_install", "opora_debit", "kedr_student"]


def oracle_answers(doc_id: str) -> dict[str, list[dict]]:
    gold = json.loads((GOLD_DIR / f"{doc_id}.json").read_text(encoding="utf-8"))
    section = {n["id"]: n["section"] for n in gold["nodes"]}
    label = {n["id"]: n["label"] for n in gold["nodes"]}
    term = {e[1]: e[2] for e in gold["edges"] if e[0] == "DEFINED_AS"}
    answers: dict[str, list[dict]] = {}
    for c in clauses(RAW_DIR / f"{doc_id}.md"):
        own = [n for n in gold["nodes"] if n["section"] == c["id"] and n["label"] != "Exception"]
        events = [e for e in gold["edges"] if e[0] in ("CANCELS", "TRIGGERS") and e[3] == c["id"]]
        items, used = [], set()
        for n in own:
            ev = next((e for e in events if e[2] == n["id"] and e[0] == "TRIGGERS"), None)
            if ev:
                used.add(tuple(ev))
            items.append({"kind": n["label"].lower(), "term": term.get(n["id"]), "event": ev[1] if ev else None,
                          "event_relation": "triggers" if ev else "none", "parent_clause": None})
        exc = [n for n in gold["nodes"] if n["section"] == c["id"] and n["label"] == "Exception"]
        if exc:
            parent = next(e[1] for e in gold["edges"] if e[0] == "HAS_EXCEPTION" and e[2] == exc[0]["id"])
            items.append({"kind": "exception", "term": None, "event": None, "event_relation": "none",
                          "parent_clause": section[parent]})
        for e in events:
            if tuple(e) in used:
                continue
            target_clause = section[e[2]]
            items.append({"kind": "none", "term": None, "event": e[1], "event_relation": e[0].lower(),
                          "parent_clause": target_clause if label[e[2]] else None})
        answers[c["id"]] = items[:MAX_ITEMS]
    return answers


def ceiling_graph(doc_id: str, catalog: dict) -> dict:
    meta, _ = parse_doc(RAW_DIR / f"{doc_id}.md")
    events = {e: [p for p in catalog[e]["aliases"].split("|") if p] for e in DOC_TYPE_EVENTS[meta["doc_type"]]}
    return build_graph(doc_id, clauses(RAW_DIR / f"{doc_id}.md"), oracle_answers(doc_id), catalog, events, Stats())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", nargs="*", default=SAMPLE, help="doc_id; all — все 20 продуктов")
    args = parser.parse_args()
    catalog = {n["id"]: n for n in json.loads((GOLD_DIR / "glossary.json").read_text(encoding="utf-8"))["nodes"]}
    docs = args.docs
    if docs == ["all"]:
        docs = [parse_doc(p)[0]["doc_id"] for p in all_docs()
                if parse_doc(p)[0]["doc_type"] not in ("glossary", "regulatory", "case")]
    with tempfile.TemporaryDirectory() as tmp:
        for d in docs:
            (Path(tmp) / f"{d}.json").write_text(json.dumps(ceiling_graph(d, catalog), ensure_ascii=False), encoding="utf-8")
        import sys
        sys.argv = ["eval_extraction", "--pred", tmp]
        eval_extraction.main()


if __name__ == "__main__":
    main()
