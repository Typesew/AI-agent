"""Прогон 20 вопросов из eval/questions.json.

    python -m eval.run_eval              # офлайн: маршрут — эталон из questions.json, «ответ» — весь контекст
    python -m eval.run_eval --llm        # с Ollama: реальный маршрутизатор и ответ Qwen3
    python -m eval.run_eval --no-graph   # то же без графа (для сравнения на защите)
    python -m eval.run_eval --only 4 9

Офлайн-режим проверяет всё, кроме формулировок LLM: распознавание, ретриверы,
граф, MCP-инструменты, числа и источники. Проходит ли текст модели — только с --llm.
Текст ответа не сравнивается дословно: проверяются обязательные факты, источники,
маршрут, вызванный инструмент и числа.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "questions.json"


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


def check(q: dict, resp) -> dict:
    text = _norm(resp.render() + "\n" + json.dumps([c.get("result") for c in resp.tool_calls], ensure_ascii=False))
    files = {s.file for s in resp.sources}
    problems = []
    expect_route = q.get("expect_route")
    if expect_route and resp.route != expect_route:
        problems.append(f"маршрут {resp.route}, ожидался {expect_route}")
    if not expect_route and resp.route in ("refuse", "nodata", "clarify"):
        problems.append(f"маршрут {resp.route}: {resp.answer[:80]}")
    called = [c["tool"] for c in resp.tool_calls]
    if q.get("tool") and q["tool"] not in called:
        problems.append(f"не вызван {q['tool']} (вызваны {called})")
    for key, want in q.get("numbers", {}).items():
        got = next((c["result"].get(key) for c in resp.tool_calls if "result" in c and key in c["result"]), None)
        ok = abs(got - want) < 1 if isinstance(want, (int, float)) and isinstance(got, (int, float)) else got == want
        if not ok:
            problems.append(f"{key}={got}, ожидалось {want}")
    missing = [f for f in q.get("facts", []) if _norm(f) not in text]
    if missing:
        problems.append(f"нет фактов: {missing}")
    lost = [f for f in q.get("sources", []) if f not in files]
    if lost:
        problems.append(f"нет источников: {lost}")
    if q.get("sources_any") and not files & set(q["sources_any"]):
        problems.append(f"нет ни одного из источников {q['sources_any']}")
    return {"id": q["id"], "type": q["type"], "passed": not problems, "problems": problems,
            "route": resp.route, "tools": called, "sources": sorted(files), "answer": resp.render(),
            "notes": resp.notes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="реальный Ollama вместо офлайн-подмены")
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--only", type=int, nargs="*")
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
            resp = agent.ask(q["question"], use_graph=not args.no_graph)
            r = check(q, resp)
            results.append(r)
            mark = "OK  " if r["passed"] else "FAIL"
            print(f"{mark} {q['id']:>2} [{q['type']:<9}] {q['question'][:70]}")
            for p in r["problems"]:
                print(f"         - {p}")
    finally:
        agent.close()

    passed = sum(r["passed"] for r in results)
    print(f"\nПройдено {passed}/{len(results)} ({passed / max(len(results), 1):.0%})")
    suffix = ("llm" if args.llm else "offline") + ("_nograph" if args.no_graph else "")
    out = ROOT / "eval" / f"results_{suffix}.json"
    out.write_text(json.dumps({"mode": suffix, "passed": passed, "total": len(results), "results": results},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Результаты: {out.relative_to(ROOT)}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
