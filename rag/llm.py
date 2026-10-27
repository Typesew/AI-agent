"""Минимальный клиент Ollama на стандартной библиотеке (без лишних зависимостей)."""

import json
import os
import urllib.error
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3")
# Держать модель в памяти между вопросами: загрузка 8B с диска на CPU — десятки секунд.
KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")


class LLMUnavailable(RuntimeError):
    """Ollama не запущен или модель не скачана."""


def available() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2)
        return True
    except (urllib.error.URLError, OSError):
        return False


def _post(path: str, body: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        f"{OLLAMA_URL}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise LLMUnavailable(f"Ollama ответил {e.code}: {e.read().decode(errors='replace')[:200]}") from e
    except (urllib.error.URLError, OSError) as e:
        raise LLMUnavailable(f"Ollama недоступен по {OLLAMA_URL}: {e}") from e


def chat(messages: list[dict], model: str = DEFAULT_MODEL, schema: dict | None = None, timeout: int = 600,
         max_tokens: int | None = None) -> str:
    """max_tokens ограничивает длину ответа: на CPU (~5 ток/с) зациклившаяся модель иначе
    генерирует до таймаута."""
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,  # для Qwen3: без рассуждений быстрее и JSON чище
        "keep_alive": KEEP_ALIVE,
        "options": {"temperature": 0, **({"num_predict": max_tokens} if max_tokens else {})},
    }
    if schema:
        body["format"] = schema  # structured output: Ollama держит ответ в рамках JSON-схемы
    return _post("/api/chat", body, timeout)["message"]["content"]


def embed(texts: list[str], model: str = EMBED_MODEL, timeout: int = 300) -> list[list[float]]:
    return _post("/api/embed", {"model": model, "input": texts}, timeout)["embeddings"]
