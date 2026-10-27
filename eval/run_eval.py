"""Прогон 20 вопросов из eval/questions.json.

    python -m eval.run_eval              # офлайн: маршрут — эталон из questions.json, «ответ» — весь контекст
    python -m eval.run_eval --llm        # с Ollama: реальный маршрутизатор и ответ Qwen3
    python -m eval.run_eval --no-graph   # то же без графа (для сравнения на защите)
    python -m eval.run_eval --only 4 9

Офлайн-режим проверяет всё, кроме формулировок LLM: распознавание, ретриверы,
граф, MCP-инструменты, числа и источники. Проходит ли текст модели — только с --llm.

Результат вопроса разделён на части (вопрос пройден, если пройдены все):
- route — маршрут и вызванный инструмент, числа результата инструмента;
- table — таблица выводов для вопросов-списков (строит код, детерминированно);
- text — текст модели: обязательные факты, источники и запреты `forbidden`.

Проверки фактов и запретов нечувствительны к форме слова (стемы из rag.text), числа
сравниваются без пробелов. Запрет-строка не должна встречаться в тексте; запрет-список
[A, B] нарушен, если A и B стоят в одном предложении без «не»/«нет» (эвристика: она
пропустит ошибку вида «A сохраняет, а B нет», если ошибочна именно A).

    python -m eval.run_eval --llm --router-cache   # повторы: не звать маршрутизатор на неизменённые вопросы
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "questions.json"
ROUTER_CACHE = ROOT / "eval" / "router_cache.json"
NEGATION = {"не", "нет", "ни"}


def _norm(s: str) -> str:
    return s.lower().replace("ё", "е").replace("\xa0", " ")


class OfflineLLM:
    """Маршрут — эталон из questions.json; «ответ» — весь контекст и ИСПОЛЬЗОВАНО по всем id."""

    def __init__(self):
        self.router_answer: dict = {}

    def __call__(self, messages, schema=None, **kw):
        if schema:
            return json.dumps({"product_hints": [], **self.router_answer}, ensure_ascii=False)
        body = messages[-1]["content"]
        ids = [line[1:line.index("]")] for line in body.splitlines() if line.startswith("[") and "]" in line]
        return body + "\nИСПОЛЬЗОВАНО: " + ",".join(ids)


def contains(text: str, phrase: str) -> bool:
    """Есть ли фраза в тексте с точностью до формы слова; числа — без пробелов."""
    from rag.text import find_spans, tokens
    if re.search(r"\d", phrase):
        return re.sub(r"\s", "", _norm(phrase)) in re.sub(r"\s", "", _norm(text))
    return _norm(phrase) in _norm(text) or bool(find_spans(phrase, tokens(text)))


def forbidden_hits(text: str, forbidden: list) -> list[str]:
    hits = []
    sentences = [x for x in re.split(r"(?<=[.!?\n])\s+|\n", text) if x.strip()]
    for item in forbidden:
        if isinstance(item, str):
            if contains(text, item):
                hits.append(item)
        elif any(all(contains(s, part) for part in item) and not NEGATION & set(_norm(s).replace("«", " ").split())
                 for s in sentences):
            hits.append(" + ".join(item))
    return hits


def check(q: dict, resp) -> dict:
    tool_json = json.dumps([c.get("result") for c in resp.tool_calls], ensure_ascii=False)
    # Текст модели: для вопроса-списка это резюме под таблицей, иначе весь ответ.
    model_text = resp.answer
    files = {s.file for s in resp.sources}
    parts = {"route": [], "table": [], "text": []}
    expect_route = q.get("expect_route")
    if expect_route and resp.route != expect_route:
        parts["route"].append(f"маршрут {resp.route}, ожидался {expect_route}")
    if not expect_route and resp.route in ("refuse", "nodata", "clarify"):
        parts["route"].append(f"маршрут {resp.route}: {resp.answer[:80]}")
    called = [c["tool"] for c in resp.tool_calls]
    if q.get("tool") and q["tool"] not in called:
        parts["route"].append(f"не вызван {q['tool']} (вызваны {called})")
    for key, want in q.get("numbers", {}).items():
        got = next((c["result"].get(key) for c in resp.tool_calls if "result" in c and key in c["result"]), None)
        ok = abs(got - want) < 1 if isinstance(want, (int, float)) and isinstance(got, (int, float)) else got == want
        if not ok:
            parts["route"].append(f"{key}={got}, ожидалось {want}")

    table = getattr(resp, "table", None)
    if "table" in q:
        got = {r.product: r.verdict for r in table.rows} if table is not None else {}
        wrong = {p: (got.get(p), v) for p, v in q["table"].items() if got.get(p) != v}
        if wrong:
            parts["table"].append(f"выводы таблицы: {wrong}")
    # Факты ищем там, где они должны быть: у вопроса-списка — в таблице, иначе — в тексте ответа.
    fact_text = (table.markdown() if table is not None else model_text) + "\n" + tool_json
    missing = [f for f in q.get("facts", []) if not contains(fact_text, f)]
    if missing:
        parts["table" if table is not None else "text"].append(f"нет фактов: {missing}")
    lost = [f for f in q.get("sources", []) if f not in files]
    if lost:
        parts["text"].append(f"нет источников: {lost}")
    if q.get("sources_any") and not files & set(q["sources_any"]):
        parts["text"].append(f"нет ни одного из источников {q['sources_any']}")
    rejected = [n for n in resp.notes if n.startswith("резюме модели отброшено")]
    if rejected:  # пользователь получил верную таблицу и резюме из кода, но текст модели не прошёл
        parts["text"].append(rejected[0])
    hits = forbidden_hits(model_text, q.get("forbidden", []))
    if hits:
        parts["text"].append(f"запрещённое утверждение: {hits}")

    problems = [p for v in parts.values() for p in v]
    return {"id": q["id"], "type": q["type"], "passed": not problems, "problems": problems,
            "parts": {k: not v for k, v in parts.items()},
            "route": resp.route, "tools": called, "sources": sorted(files), "answer": resp.render(),
            "notes": resp.notes}


class RouterCache:
    """Кэш ответов маршрутизатора по полному тексту промпта: повторный прогон не зовёт
    модель на неизменённые вопросы. Ответы самой модели (не маршрутизатора) не кэшируются."""

    def __init__(self, chat, path: Path = ROUTER_CACHE):
        self.chat, self.path = chat, path
        self.data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.hits = 0

    def __call__(self, messages, schema=None, **kw):
        if not schema:
            return self.chat(messages, **kw)
        key = hashlib.sha256(json.dumps([messages, schema], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if key in self.data:
            self.hits += 1
            return self.data[key]
        out = self.chat(messages, schema=schema, **kw)
        self.data[key] = out
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="реальный Ollama вместо офлайн-подмены")
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--only", type=int, nargs="*")
    parser.add_argument("--router-cache", action="store_true", help="кэшировать ответы маршрутизатора (для повторов)")
    args = parser.parse_args()

    from agent.agent import Agent
    from agent.router import Router
    from rag import llm

    if args.llm and not llm.available():
        parser.error(f"Ollama недоступен по {llm.OLLAMA_URL}")
    questions = [q for q in json.loads(QUESTIONS.read_text(encoding="utf-8")) if not args.only or q["id"] in args.only]

    # Файлы save_summary из прогона не засоряют data/summaries.
    os.environ.setdefault("BANK_SUMMARY_DIR", tempfile.mkdtemp(prefix="eval_summaries_"))
    offline = None if args.llm else OfflineLLM()
    chat = llm.chat if args.llm else offline
    if args.llm and args.router_cache:
        chat = RouterCache(chat)
    agent = Agent(router=Router(chat=chat), chat=chat)
    if agent.retriever.vector.note:
        print("ВНИМАНИЕ:", agent.retriever.vector.note)
    print(f"Режим: {'Ollama ' + llm.DEFAULT_MODEL if args.llm else 'офлайн (эталонный маршрут, ответ = контекст)'}; "
          f"граф: {'выкл' if args.no_graph else 'вкл'}; поиск: {agent.retriever.vector.mode}\n")

    results = []
    try:
        for q in questions:
            if offline:
                offline.router_answer = q["router"]
            started = time.time()
            resp = agent.ask(q["question"], use_graph=not args.no_graph)
            r = check(q, resp)
            r["seconds"] = round(time.time() - started, 1)
            results.append(r)
            mark = "OK  " if r["passed"] else "FAIL"
            print(f"{mark} {q['id']:>2} [{q['type']:<9}] {q['question'][:62]}  ({r['seconds']} с)", flush=True)
            for p in r["problems"]:
                print(f"         - {p}")
    finally:
        agent.close()

    passed = sum(r["passed"] for r in results)
    print(f"\nПройдено {passed}/{len(results)} ({passed / max(len(results), 1):.0%})")
    for part, label in (("route", "маршрут, инструмент, числа"), ("table", "таблица (код)"), ("text", "текст модели")):
        print(f"  {label}: {sum(r['parts'][part] for r in results)}/{len(results)}")
    print(f"  время: всего {sum(r['seconds'] for r in results):.0f} с, "
          f"в среднем {sum(r['seconds'] for r in results) / max(len(results), 1):.1f} с на вопрос")
    suffix = ("llm" if args.llm else "offline") + ("_nograph" if args.no_graph else "") + (
        "_only_" + "_".join(map(str, args.only)) if args.only else "")
    out = ROOT / "eval" / f"results_{suffix}.json"
    out.write_text(json.dumps({"mode": suffix, "passed": passed, "total": len(results), "results": results},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Результаты: {out.relative_to(ROOT)}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
