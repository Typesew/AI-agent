"""Сравнение LLM-экстракции с эталонной разметкой по связям.

    python -m eval.eval_extraction [--pred data/extracted/llm]

Две метрики, обе печатаются:

1. Мягкая (исходная, не менялась): id у LLM свои, поэтому связь сравнивается
   по (тип, имя откуда, имя куда) с нечётким совпадением имён. Узлы продукта
   и справочника совпадают по id.
2. Строгая (добавлена позже, отдельно): связь = (тип, пункт-источник, пункт или id
   справочника у приёмника). Узел определяется пунктом договора, имена не важны,
   поэтому похожее название на чужом пункте не засчитывается. Сравнение — мультимножеством:
   два исключения в одном пункте — две связи.

Строгая метрика печатается с разбивкой по типам связей и отдельно для документов,
написанных свободным стилем (в YAML помечены «Свободные формулировки»), и шаблонных.
"""

import argparse
import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from indexing.chunk import all_docs, parse_doc

ROOT = Path(__file__).resolve().parent.parent
GOLD_DIR = ROOT / "data" / "extracted" / "gold"
NAME_THRESHOLD = 0.6
PRODUCTS_YAML = ROOT / "corpus" / "products"


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


def strict_edges(data: dict, doc_id: str) -> Counter:
    """Связи в виде (тип, пункт или id, пункт или id): узел документа заменяется номером пункта."""
    where = {n["id"]: n["section"] for n in data["nodes"]}

    def end(x: str) -> str:
        if x == doc_id:
            return "product"
        return f"п.{where[x]}" if x in where else x  # id справочника (term_*, ev_*) остаётся как есть
    return Counter((e[0], end(e[1]), end(e[2])) for e in data["edges"])


def strict_score(gold: Counter, pred: Counter) -> tuple[int, int, int]:
    return sum((gold & pred).values()), sum(pred.values()), sum(gold.values())


def free_style_docs() -> set[str]:
    return {p.stem for p in PRODUCTS_YAML.glob("*.yaml") if "Свободные формулировки" in p.read_text(encoding="utf-8")}


def _prf(tp: int, n_pred: int, n_gold: int) -> str:
    p, r = tp / max(n_pred, 1), tp / max(n_gold, 1)
    return f"{p:>6.2f}{r:>6.2f}{2 * p * r / max(p + r, 1e-9):>6.2f}   ({tp} / {n_pred} / {n_gold})"


def strict_report(pred_dir: Path) -> dict:
    """Строгая метрика по всем документам, по типам связей и по стилю документа."""
    free = free_style_docs()
    per_doc, by_rel, by_style = {}, {}, {"свободный": [0, 0, 0], "шаблонный": [0, 0, 0]}
    for path in all_docs():
        meta, _ = parse_doc(path)
        doc_id = meta["doc_id"]
        pred_path = pred_dir / f"{doc_id}.json"
        if meta["doc_type"] in ("glossary", "regulatory", "case") or not pred_path.exists():
            continue
        gold = strict_edges(json.loads((GOLD_DIR / f"{doc_id}.json").read_text(encoding="utf-8")), doc_id)
        pred = strict_edges(json.loads(pred_path.read_text(encoding="utf-8")), doc_id)
        per_doc[doc_id] = strict_score(gold, pred)
        style = by_style["свободный" if doc_id in free else "шаблонный"]
        for i, x in enumerate(per_doc[doc_id]):
            style[i] += x
        for rel in {k[0] for k in gold} | {k[0] for k in pred}:
            g = Counter({k: v for k, v in gold.items() if k[0] == rel})
            pr = Counter({k: v for k, v in pred.items() if k[0] == rel})
            acc = by_rel.setdefault(rel, [0, 0, 0])
            for i, x in enumerate(strict_score(g, pr)):
                acc[i] += x
    return {"per_doc": per_doc, "by_rel": by_rel, "by_style": by_style}


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

    rep = strict_report(args.pred)
    print("\nСтрогая метрика: связь = (тип, пункт-источник, пункт или id справочника)")
    print(f"{'документ':<20}{'P':>6}{'R':>6}{'F1':>6}   (верных / предсказано / в эталоне)")
    for doc_id, t in rep["per_doc"].items():
        print(f"{doc_id:<20}{_prf(*t)}")
    total = [sum(t[i] for t in rep["per_doc"].values()) for i in range(3)]
    print(f"{'ИТОГО':<20}{_prf(*total)}")
    print("\nПо стилю документа (строгая):")
    for style, t in rep["by_style"].items():
        if t[2]:
            print(f"{style:<20}{_prf(*t)}")
    print("\nПо типам связей (строгая):")
    for rel, t in sorted(rep["by_rel"].items()):
        print(f"{rel:<20}{_prf(*t)}")


if __name__ == "__main__":
    main()
