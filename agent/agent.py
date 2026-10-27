"""Агент: распознавание сущностей -> маршрут -> инструмент (MCP) -> контекст -> LLM -> источники."""

from dataclasses import dataclass, field

from agent.client import MCPClient, ToolCallError
from agent.router import Route, Router
from rag import llm
from rag.entities import EntityRecognizer
from rag.graph_retriever import to_dot
from rag.prompt import (LIST_MAX_TOKENS, Source, build_list_messages, build_messages, collect_sources, context_items,
                        parse_answer, summary_violations, table_sources)
from rag.retriever import Context, HybridRetriever
from rag.verdicts import VerdictBuilder, Verdicts, detect

REFUSAL = ("Я отвечаю только на вопросы об условиях банковских карт, кредитов и рассрочек из учебного корпуса. "
           "С этим вопросом помочь не могу.")
NO_DATA = "В корпусе нет данных по этому вопросу."
# Банковские темы, которых в корпусе нет: на них честное «нет данных», а не выдумка.
NO_DATA_STEMS = ["вклад", "депозит", "ипотек", "автокредит", "страхов", "инвест", "брокер", "облигац", "пенсионн"]
# Длина обычного ответа: «Плюсы / Минусы / Подводные камни / Итог» укладываются в ~400–600 токенов;
# без лимита 8B на CPU (~5 ток/с) иногда пишет дольше нужного.
ANSWER_MAX_TOKENS = 700
# Фрагментов документов в контексте ответа: 3 вместо 5 — короче промпт, быстрее на CPU;
# факты графа идут отдельно. Проверено офлайн-оценкой (20/20 с k=3).
CONTEXT_CHUNKS = 3


@dataclass
class AgentResponse:
    answer: str
    route: str
    sources: list[Source] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)  # {tool, args, result | error}
    context: Context | None = None
    notes: list[str] = field(default_factory=list)
    table: Verdicts | None = None  # вопрос-список: выводы по продуктам посчитал код

    @property
    def dot(self) -> str:
        if self.table is not None:
            return to_dot(self.table.facts()) if self.table.facts() else ""
        return self.context.dot if self.context else ""

    def render(self) -> str:
        out = self.answer
        if self.table is not None:
            out = self.table.markdown() + "\n\n" + out
        if self.sources:
            out += "\n\n**Источники**\n" + "\n".join(f"- {s.render()}" for s in self.sources)
        return out


class Agent:
    def __init__(self, retriever: HybridRetriever | None = None, router: Router | None = None,
                 tools: MCPClient | None = None, chat=None):
        self.retriever = retriever or HybridRetriever()
        self.recognizer: EntityRecognizer = self.retriever.recognizer
        self.router = router or Router(chat=chat)
        self.chat = chat or (lambda *a, **k: llm.chat(*a, **k))
        self._tools = tools
        graph = self.retriever.graph
        self.verdicts = VerdictBuilder(graph) if graph is not None else None

    @property
    def tools(self) -> MCPClient:
        if self._tools is None:  # подпроцесс поднимаем при первом расчёте
            self._tools = MCPClient()
        return self._tools

    def close(self):
        if self._tools is not None:
            self._tools.close()

    def ask(self, question: str, use_graph: bool = True) -> AgentResponse:
        ent = self.recognizer.recognize(question)
        if not ent.domain and not ent.numbers:
            return AgentResponse(REFUSAL, "refuse", notes=["нет слов предметной области и чисел"])
        if ent.is_empty() and any(s in question.lower() for s in NO_DATA_STEMS):
            return AgentResponse(NO_DATA, "nodata", notes=["тема вне корпуса"])

        if use_graph and self.verdicts is not None and detect(ent) and not ent.numbers:
            return self._list_answer(question, ent)

        route = self.router.route(question, ent)
        notes = [route.note] if route.note else []
        if route.route == "refuse":
            return AgentResponse(REFUSAL, "refuse", notes=notes)
        if route.clarify:
            return AgentResponse(route.clarify, "clarify", notes=notes)

        calls = []
        if route.tool and route.tool != "save_summary":
            call = {"tool": route.tool, "args": route.args}
            try:
                call["result"] = self.tools.call(route.tool, route.args)
            except ToolCallError as e:
                return AgentResponse(f"Не получилось посчитать: {e}", "clarify", tool_calls=[{**call, "error": str(e)}])
            calls.append(call)

        ctx = (self.retriever.retrieve(question, ent, use_graph=use_graph, k=CONTEXT_CHUNKS)
               if route.route in ("rag", "both")
               else Context(ent))
        if route.route == "rag" and ctx.is_empty():
            return AgentResponse(NO_DATA, "nodata", context=ctx, notes=notes)

        tool_results = [c for c in calls if "result" in c]
        items = context_items(ctx, tool_results)
        raw = self.chat(build_messages(question, ctx, tool_results), max_tokens=ANSWER_MAX_TOKENS)
        answer, used, parsed = parse_answer(raw, items)
        if not parsed:
            notes.append("строка ИСПОЛЬЗОВАНО не разобрана — показаны все источники контекста")
        sources = [] if answer.strip().startswith(NO_DATA) else collect_sources(used, items, answer)
        for c in tool_results:
            sources += [Source(s["file"], f"п. {s['section']}", s["date"], s["file"])
                        for s in c["result"].get("sources", []) if (s["file"], f"п. {s['section']}") not in
                        {(x.file, x.section) for x in sources}]

        resp = AgentResponse(answer, route.route, sources, calls, ctx, notes)
        if route.tool == "save_summary":
            call = {"tool": "save_summary", "args": {**route.args, "text": resp.render()}}
            try:
                call["result"] = self.tools.call("save_summary", call["args"])
                resp.answer += f"\n\nРазбор сохранён в файл `{call['result']['path']}`."
            except ToolCallError as e:
                call["error"] = str(e)
                resp.answer += f"\n\nСохранить файл не удалось: {e}"
            resp.tool_calls.append(call)
        return resp

    def _list_answer(self, question: str, ent) -> AgentResponse:
        """Вопрос-список («какие карты…», «по каким рассрочкам…»): таблицу строит код по графу.

        Маршрутизатор не нужен: вопрос распознан по сущностям, расчёта в нём нет. Модель видит
        только таблицу (без фрагментов документов и посторонних фактов — комиссий, кэшбэка)
        и пишет короткое резюме; если в резюме продукт или число не из таблицы, резюме
        заменяется собранным кодом.
        """
        table = self.verdicts.build(ent)
        raw = self.chat(build_list_messages(question, table), max_tokens=LIST_MAX_TOKENS)
        summary, _, _ = parse_answer(raw, {})
        notes = ["таблица выводов построена кодом по графу"]
        problems = summary_violations(summary, table)
        if problems:
            notes.append("резюме модели отброшено: " + "; ".join(problems))
            summary = table.summary()
        return AgentResponse(summary, "list", table_sources(table), context=Context(ent), notes=notes, table=table)
