"""YAML-корпус -> markdown-документы, эталонный граф и каталог сущностей.

    python -m corpus.build_corpus

Один источник истины: из каждого пункта YAML получаются и строка документа,
и узел графа с номером этого пункта, поэтому документ и разметка не расходятся.

Выход:
- data/raw/*.md                 — документы с front-matter (продукты, глоссарий, extra);
- data/extracted/gold/*.json    — эталонная разметка в формате indexing.build_graph;
- data/catalog.json             — банки, продукты, термины, события для rag.entities и инструментов.
"""

import json
import shutil
from pathlib import Path

import yaml

from indexing.schema import check_edge

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
RAW_DIR = ROOT / "data" / "raw"
GOLD_DIR = ROOT / "data" / "extracted" / "gold"
CATALOG = ROOT / "data" / "catalog.json"

DISCLAIMER = "_Учебный документ, не оферта. Банк и условия вымышлены._"
TITLES = {
    "credit_card": "Условия кредитной карты «{}»",
    "installment": "Условия рассрочки «{}»",
    "debit_card": "Условия дебетовой карты «{}»",
    "student_card": "Условия студенческой карты «{}»",
}
REL_FOR_LABEL = {"Condition": "HAS_CONDITION", "Fee": "CHARGES"}
BANK_ALIASES = {
    "bank_sever": "северный путь|северном пути",
    "bank_opora": "точка опоры|точке опоры",
    "bank_merid": "меридиан",
    "bank_kedr": "кедр",
    "bank_balt": "балтийский|балтийском",
}


def front_matter(meta: dict) -> str:
    keys = ["doc_id", "bank_id", "bank", "product", "product_term", "doc_type", "date", "aliases"]
    return "---\n" + "\n".join(f"{k}: {meta.get(k, '-')}" for k in keys) + "\n---\n"


def load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["date"] = str(data["date"])  # YAML читает 2026-09-01 как date
    return data


def load_products() -> list[dict]:
    return [load_yaml(p) for p in sorted((CORPUS / "products").glob("*.yaml"))]


def product_doc(p: dict) -> tuple[str, dict]:
    """Возвращает markdown и эталонную разметку одного продукта."""
    pid = p["doc_id"]
    lines = [front_matter({**p, "aliases": "|".join(p.get("aliases", []))}),
             f"# {TITLES[p['doc_type']].format(p['product'])}", "", DISCLAIMER, ""]
    nodes, edges = [], []
    for section in p["sections"]:
        lines.append(f"## {section['number']}. {section['title']}")
        for clause in section["clauses"]:
            cid = clause["id"]
            if not cid.startswith(f"{section['number']}."):
                raise ValueError(f"{pid}: пункт {cid} не в разделе {section['number']}")
            lines.append(f"{cid}. {clause['text']}")
            for n in clause.get("nodes", []) + ([clause["node"]] if "node" in clause else []):
                nodes.append({"id": n["id"], "label": n["label"], "name": n["name"], "value": n.get("value", ""),
                              "aliases": "", "section": cid, **({"scope": "|".join(n["scope"])} if "scope" in n else {})})
                if n["label"] == "Exception":
                    edges.append(["HAS_EXCEPTION", n["of"], n["id"], cid])
                    if "event" in n:
                        edges.append(["APPLIES_TO", n["id"], n["event"], cid])
                else:
                    edges.append([REL_FOR_LABEL[n["label"]], pid, n["id"], cid])
                if "term" in n:
                    edges.append(["DEFINED_AS", n["id"], n["term"], cid])
        lines.append("")
    for e in p.get("events", []):
        edges.append([e["rel"], e["event"], e["target"], e["section"]])
    return "\n".join(lines), {"nodes": nodes, "edges": edges}


def glossary_doc(g: dict) -> tuple[str, dict]:
    meta = {"doc_id": "glossary", "doc_type": "glossary", "date": g["date"]}
    lines = [front_matter(meta), "# Глоссарий банковских терминов", "", DISCLAIMER, "", "## 1. Термины"]
    nodes = []
    for i, t in enumerate(g["terms"], 1):
        lines.append(f"1.{i}. {t['name']} — {t['definition']}")
        nodes.append({"id": t["id"], "label": "Term", "name": t["name"], "value": t["definition"],
                      "aliases": t["aliases"], "section": f"1.{i}"})
    lines += ["", "## 2. События, которые меняют условия"]
    for i, e in enumerate(g["events"], 1):
        lines.append(f"2.{i}. {e['name']} — {e['definition']}")
        nodes.append({"id": e["id"], "label": "Event", "name": e["name"], "value": e["definition"],
                      "aliases": e["aliases"], "section": f"2.{i}"})
    return "\n".join(lines) + "\n", {"nodes": nodes, "edges": []}


def validate(products: list[dict], gold: dict[str, dict], glossary: dict) -> list[str]:
    """Проверяет ссылки и схему до записи файлов: ошибка в YAML не должна дойти до графа."""
    labels = {n["id"]: n["label"] for g in gold.values() for n in g["nodes"]}
    labels.update({p["doc_id"]: "Product" for p in products})
    errors = []
    seen: set[str] = set()
    for doc_id, g in gold.items():
        for n in g["nodes"]:
            if n["id"] in seen:
                errors.append(f"{doc_id}: повтор id {n['id']}")
            seen.add(n["id"])
        for rel, a, b, _ in g["edges"]:
            err = check_edge(rel, labels.get(a), labels.get(b))
            if err:
                errors.append(f"{doc_id}: {rel} {a} -> {b}: {err}")
    for p in products:
        if p["product_term"] not in {t["id"] for t in glossary["terms"]}:
            errors.append(f"{p['doc_id']}: нет термина {p['product_term']}")
    return errors


def build_catalog(products: list[dict], glossary: dict) -> dict:
    banks = {}
    for p in products:
        banks.setdefault(p["bank_id"], {"id": p["bank_id"], "name": p["bank"], "aliases": BANK_ALIASES[p["bank_id"]]})
    return {
        "banks": list(banks.values()),
        "products": [{
            "id": p["doc_id"], "name": p["product"], "bank_id": p["bank_id"], "bank": p["bank"],
            "type_term": p["product_term"], "doc_type": p["doc_type"], "date": p["date"],
            "aliases": "|".join(p.get("match_phrases") or [p["product"].lower()] + p.get("aliases", [])),
            "require_context": p.get("require_context", []), "tool_params": p.get("tool_params", {}),
        } for p in products],
        "terms": [{k: t[k] for k in ("id", "name", "aliases", "definition")} for t in glossary["terms"]],
        "events": [{k: e[k] for k in ("id", "name", "aliases", "definition")} for e in glossary["events"]],
        "product_types": sorted({p["product_term"] for p in products}),
    }


def main():
    products = load_products()
    glossary = load_yaml(CORPUS / "glossary.yaml")
    docs, gold = {}, {}
    for p in products:
        docs[p["doc_id"]], gold[p["doc_id"]] = product_doc(p)
    docs["glossary"], gold["glossary"] = glossary_doc(glossary)

    errors = validate(products, gold, glossary)
    if errors:
        raise SystemExit("Ошибки в корпусе:\n" + "\n".join(errors))

    for d in (RAW_DIR, GOLD_DIR):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    for doc_id, text in docs.items():
        (RAW_DIR / f"{doc_id}.md").write_text(text, encoding="utf-8")
        (GOLD_DIR / f"{doc_id}.json").write_text(json.dumps(gold[doc_id], ensure_ascii=False, indent=1), encoding="utf-8")
    extra = sorted((CORPUS / "extra").glob("*.md"))
    for path in extra:
        shutil.copy(path, RAW_DIR / path.name)
    CATALOG.write_text(json.dumps(build_catalog(products, glossary), ensure_ascii=False, indent=1), encoding="utf-8")

    n_nodes = sum(len(g["nodes"]) for g in gold.values())
    n_edges = sum(len(g["edges"]) for g in gold.values())
    print(f"Продуктов {len(products)}, глоссарий 1, extra {len(extra)}; эталон: узлов {n_nodes}, связей {n_edges}")


if __name__ == "__main__":
    main()
