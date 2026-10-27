"""Сравнение LLM-экстракции с эталонной разметкой по связям.

    python -m eval.eval_extraction [--pred data/extracted/llm]

id у LLM свои, поэтому связь сравнивается по (тип, имя откуда, имя куда)
с нечётким совпадением имён. Узлы продукта и справочника совпадают по id.
"""

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path

from indexing.chunk import all_docs, parse_doc

ROOT = Path(__file__).resolve().parent.parent
GOLD_DIR = ROOT / "data" / "extracted" / "gold"
NAME_THRESHOLD = 0.6


def _similar(a: str, b: str) -> bool:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= NAME_THRESHOLD


def _edges(data: dict, catalog: dict[str, str], product: tuple[str, str]) -> list[tuple[str, str, str]]:
    names = {**catalog, product[0]: product[1], **{n["id"]: n["name"] for n in data["nodes"]}}
    return [(e[0], names.get(e[1], e[1]), names.get(e[2], e[2])) for e in data["edges"]]


def score(gold: list[tuple], pred: list[tuple]) -> tuple[int, int, int]:
    unmatched = list(gold)
    tp = 0
    for rel, a, b in pred:
        hit = next((g for g in unmatched if g[0] == rel and _similar(g[1], a) and _similar(g[2], b)), None)
        if hit:
            unmatched.remove(hit)
            tp += 1
    return tp, len(pred), len(gold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=Path, default=ROOT / "data" / "extracted" / "llm")
    args = parser.parse_args()

    glossary = json.loads((GOLD_DIR / "glossary.json").read_text(encoding="utf-8"))
    catalog = {n["id"]: n["name"] for n in glossary["nodes"]}
    total = [0, 0, 0]
    print(f"{'документ':<20}{'P':>6}{'R':>6}{'F1':>6}   (верных / предсказано / в эталоне)")
    for path in all_docs():
        meta, _ = parse_doc(path)
        doc_id = meta["doc_id"]
        pred_path = args.pred / f"{doc_id}.json"
        if meta["doc_type"] in ("glossary", "regulatory", "case") or not pred_path.exists():
            continue
        product = (doc_id, meta["product"])
        gold = _edges(json.loads((GOLD_DIR / f"{doc_id}.json").read_text(encoding="utf-8")), catalog, product)
        pred = _edges(json.loads(pred_path.read_text(encoding="utf-8")), catalog, product)
        tp, n_pred, n_gold = score(gold, pred)
        total = [t + x for t, x in zip(total, (tp, n_pred, n_gold))]
        p, r = tp / max(n_pred, 1), tp / max(n_gold, 1)
        print(f"{doc_id:<20}{p:>6.2f}{r:>6.2f}{2 * p * r / max(p + r, 1e-9):>6.2f}   ({tp} / {n_pred} / {n_gold})")
    tp, n_pred, n_gold = total
    p, r = tp / max(n_pred, 1), tp / max(n_gold, 1)
    print(f"{'ИТОГО':<20}{p:>6.2f}{r:>6.2f}{2 * p * r / max(p + r, 1e-9):>6.2f}   ({tp} / {n_pred} / {n_gold})")


if __name__ == "__main__":
    main()
