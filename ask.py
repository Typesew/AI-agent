"""CLI для проверки графового пути без интерфейса.

    python ask.py "Что отменяет льготный период по карте «Полярная 120»?"
    python ask.py --demo                # прогон демо-вопросов
    python ask.py "..." --mermaid       # цепочка из графа для слайдов
    python ask.py "..." --agent         # полный ответ агента (нужен Ollama)
"""

import argparse

from rag.graph_retriever import GraphRetriever, to_mermaid

DEMO = [
    "Что такое льготный период и когда он пропадает?",
    "Что отменяет льготный период по карте «Полярная 120»?",
    "Какие операции не участвуют в кэшбэке по «Полярная 120»?",
    "Какие кредитные карты сохраняют льготный период при снятии наличных?",
    "По каким рассрочкам при просрочке начисляются проценты?",
    "Сравни по обслуживанию и снятию наличных кредитки «Опорная 55» и «Меридиан 150».",
    "Какие студенческие карты после потери статуса могут стать платными?",
    "Какая погода завтра?",
]


def show_graph(retriever: GraphRetriever, question: str, mermaid: bool):
    print(f"\n=== {question}")
    r = retriever.retrieve(question)
    if r.entities.is_empty():
        print("(сущностей предметной области не найдено)")
        return
    print("Найдено:", r.entities.summary())
    for f in r.facts:
        src = f" [{f.source['file']}, п. {f.source['section']}, ред. {f.source['date']}]" if f.source else ""
        print(f"- {f.text}{src}")
    if r.table:
        print("\n" + r.table)
    if mermaid and r.facts:
        print("\n" + to_mermaid(r.facts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--mermaid", action="store_true")
    parser.add_argument("--agent", action="store_true", help="ответ агента целиком (Ollama)")
    args = parser.parse_args()
    questions = DEMO if args.demo else [args.question or parser.error("нужен вопрос или --demo")]

    if args.agent:
        from agent.agent import Agent
        agent = Agent()
        try:
            for q in questions:
                print(f"\n=== {q}\n{agent.ask(q).render()}")
        finally:
            agent.close()
        return

    retriever = GraphRetriever()
    for q in questions:
        show_graph(retriever, q, args.mermaid)


if __name__ == "__main__":
    main()
