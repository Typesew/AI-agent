"""Чанки всех документов -> эмбеддинги -> Chroma.

    python -m indexing.embed_index                    # bge-m3 через Ollama (рабочий режим)
    python -m indexing.embed_index --embedder hash    # без Ollama, только проверка обвязки

Эмбеддинги считаем сами и передаём в Chroma явно, чтобы Chroma не скачивала
свою модель. Имя эмбеддера пишется в метаданные коллекции: ретривер обязан
считать вектор запроса той же моделью.
"""

import argparse
import hashlib
import math
from pathlib import Path

import chromadb
from chromadb.config import Settings

from indexing.chunk import all_docs, chunk_doc
from rag import llm

ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION = "bank_chunks"
BATCH = 16
HASH_DIM = 512


def hash_embed(texts: list[str]) -> list[list[float]]:
    """Детерминированный «мешок стемов» в фиксированной размерности.

    Не семантика, а заглушка: позволяет проверить индекс, фильтры и RRF без Ollama.
    """
    from rag.vector_retriever import bm25_terms  # здесь, чтобы не было циклического импорта

    out = []
    for text in texts:
        vec = [0.0] * HASH_DIM
        for t in bm25_terms(text):
            vec[int(hashlib.md5(t.encode()).hexdigest(), 16) % HASH_DIM] += 1.0
        n = math.sqrt(sum(v * v for v in vec)) or 1.0
        out.append([v / n for v in vec])
    return out


def embedder(name: str):
    return hash_embed if name == "hash" else (lambda texts: llm.embed(texts, model=name))


def client(path: Path = CHROMA_DIR):
    return chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))


def build(embedder_name: str = llm.EMBED_MODEL, path: Path = CHROMA_DIR) -> int:
    chunks = [c for p in all_docs() for c in chunk_doc(p)]
    embed = embedder(embedder_name)
    db = client(path)
    try:
        db.delete_collection(COLLECTION)
    except Exception:  # коллекции ещё нет
        pass
    col = db.create_collection(COLLECTION, configuration={"hnsw": {"space": "cosine"}},
                               metadata={"embedder": embedder_name}, embedding_function=None)
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        col.add(ids=[c["id"] for c in batch], documents=[c["text"] for c in batch],
                metadatas=[c["metadata"] for c in batch], embeddings=embed([c["text"] for c in batch]))
    print(f"Chroma: {len(chunks)} чанков, эмбеддер {embedder_name}, {path}")
    return len(chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", default=llm.EMBED_MODEL, help="модель Ollama (bge-m3) или hash")
    args = parser.parse_args()
    if args.embedder != "hash" and not llm.available():
        parser.error(f"Ollama недоступен по {llm.OLLAMA_URL}; запустите `ollama serve` или используйте --embedder hash")
    build(args.embedder)


if __name__ == "__main__":
    main()
