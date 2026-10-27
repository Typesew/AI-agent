"""Агент: распознавание сущностей -> маршрут -> инструмент (MCP) -> контекст -> LLM -> источники."""

from dataclasses import dataclass, field

from agent.client import MCPClient, ToolCallError
from agent.router import Route, Router
from rag import llm
from rag.entities import EntityRecognizer
from rag.prompt import Source, build_messages, collect_sources, context_items, parse_answer
from rag.retriever import Context, HybridRetriever

REFUSAL = ("Я отвечаю только на вопросы об условиях банковских карт, кредитов и рассрочек из учебного корпуса. "
           "С этим вопросом помочь не могу.")
NO_DATA = "В корпусе нет данных по этому вопросу."
# Банковские темы, которых в корпусе нет: на них честное «нет данных», а не выдумка.
NO_DATA_STEMS = ["вклад", "депозит", "ипотек", "автокредит", "страхов", "инвест", "брокер", "облигац", "пенсионн"]


@dataclass
class AgentResponse:
    answer: str
    route: str
    sources: list[Source] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)  # {tool, args, result | error}
    context: Context | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def dot(self) -> str:
        return self.context.dot if self.context else ""

    def render(self) -> str:
        out = self.answer
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

        ctx = (self.retriever.retrieve(question, ent, use_graph=use_graph) if route.route in ("rag", "both")
               else Context(ent))
        if route.route == "rag" and ctx.is_empty():
            return AgentResponse(NO_DATA, "nodata", context=ctx, notes=notes)

        tool_results = [c for c in calls if "result" in c]
        items = context_items(ctx, tool_results)
        raw = self.chat(build_messages(question, ctx, tool_results))
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
