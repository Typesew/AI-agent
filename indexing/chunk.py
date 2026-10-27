"""Парсинг markdown-документов с front-matter и чанкинг по разделам договора."""

import re
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
MAX_CHUNK = 1200  # длиннее режем по пунктам, чтобы пункт не разрывался посередине
TARGET_CHUNK = 800

SECTION_RE = re.compile(r"^##\s+(\d+)\.\s*(.+)$")
CLAUSE_RE = re.compile(r"^\d+\.\d+\.")


def parse_doc(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    meta = {}
    if text.startswith("---"):
        header, text = text[3:].split("\n---", 1)
        for line in header.strip().splitlines():
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    meta["source_file"] = path.name
    return meta, text.strip()


def _split_clauses(lines: list[str]) -> list[list[str]]:
    """Группирует пункты раздела в куски до TARGET_CHUNK символов."""
    groups, current, size = [], [], 0
    for line in lines:
        if current and CLAUSE_RE.match(line) and size + len(line) > TARGET_CHUNK:
            groups.append(current)
            current, size = [], 0
        current.append(line)
        size += len(line)
    if current:
        groups.append(current)
    return groups


def chunk_doc(path: Path) -> list[dict]:
    meta, body = parse_doc(path)
    title = next((l[2:].strip() for l in body.splitlines() if l.startswith("# ")), meta.get("product", ""))
    sections: list[tuple[str, str, list[str]]] = []
    for line in body.splitlines():
        m = SECTION_RE.match(line)
        if m:
            sections.append((m.group(1), m.group(2), []))
        elif sections and line.strip():
            sections[-1][2].append(line.strip())

    chunks = []
    for number, name, lines in sections:
        text = "\n".join(lines)
        groups = [lines] if len(text) <= MAX_CHUNK else _split_clauses(lines)
        for i, group in enumerate(groups):
            chunks.append({
                "id": f"{meta['doc_id']}#{number}" + (f".{i}" if len(groups) > 1 else ""),
                "text": f"{title}\n{number}. {name}\n" + "\n".join(group),
                "metadata": {
                    "doc_id": meta["doc_id"],
                    "bank": meta.get("bank", ""),
                    "product": meta.get("product", ""),
                    "doc_type": meta.get("doc_type", ""),
                    "section": f"{number}. {name}",
                    "source_file": meta["source_file"],
                    "date": meta.get("date", ""),
                    "page": 1,  # у markdown нет страниц; для PDF сюда пойдёт номер страницы
                },
            })
    return chunks


def all_docs() -> list[Path]:
    return sorted(RAW_DIR.glob("*.md"))


if __name__ == "__main__":
    for path in all_docs():
        for chunk in chunk_doc(path):
            print(chunk["id"], "|", chunk["metadata"]["section"], "|", len(chunk["text"]), "симв.")
