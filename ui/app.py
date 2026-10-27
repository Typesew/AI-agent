"""Веб-интерфейс «Банковский переводчик».

    streamlit run ui/app.py        # http://localhost:8501; MCP-сервер стартует подпроцессом

Streamlit выбран как один процесс на Python: чат из коробки, st.graphviz_chart
рисует граф локально (без CDN), минимум зависимостей.
"""

import atexit
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.agent import Agent  # noqa: E402
from rag import llm  # noqa: E402

st.set_page_config(page_title="Банковский переводчик", page_icon="🏦", layout="centered")


@st.cache_resource(show_spinner="Загружаю индексы и граф…")
def get_agent() -> Agent:
    agent = Agent()
    atexit.register(agent.close)  # остановить MCP-подпроцесс вместе с сервером Streamlit
    return agent


def show_response(data: dict):
    st.markdown(data["answer"])
    if data["sources"]:
        st.markdown("**Источники**\n" + "\n".join(f"- {s}" for s in data["sources"]))
    for call in data["tool_calls"]:
        with st.expander(f"Показать расчёт: {call['tool']}"):
            st.caption("Аргументы (проверены кодом)")
            st.json(call["args"])
            if "error" in call:
                st.error(call["error"])
            else:
                st.caption("Результат инструмента")
                st.json(call["result"])
    if data["dot"]:
        with st.expander("Цепочка в графе"):
            st.graphviz_chart(data["dot"], width="stretch")
    if data["notes"]:
        st.caption("; ".join(data["notes"]))


def to_data(resp) -> dict:
    return {"answer": resp.answer, "sources": [s.render() for s in resp.sources], "tool_calls": resp.tool_calls,
            "dot": resp.dot, "notes": resp.notes, "route": resp.route}


st.title("🏦 Банковский переводчик")
st.info("Учебный проект: банки и условия вымышлены, документы — не оферта. "
        "В каждом ответе указана дата редакции документа.", icon="ℹ️")

with st.sidebar:
    use_graph = st.toggle("Граф знаний", value=True, help="Выключите, чтобы сравнить с чистым векторным поиском")
    st.caption(f"Модель: {llm.DEFAULT_MODEL}, эмбеддинги: {llm.EMBED_MODEL}")
    if not llm.available():
        st.warning(f"Ollama недоступен по {llm.OLLAMA_URL}. Запустите `ollama serve`.")
    if st.button("Очистить чат"):
        st.session_state.messages = []
    st.markdown("**Примеры**\n- Что отменяет льготный период по карте «Полярная 120»?\n"
                "- Кредит 100 000 ₽ на 2 года под 18%, какая переплата?\n"
                "- Какие кредитные карты сохраняют льготный период при снятии наличных?")

agent = get_agent()
if agent.retriever.vector.note:
    st.sidebar.warning(agent.retriever.vector.note)

st.session_state.setdefault("messages", [])
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            show_response(msg["data"])

if question := st.chat_input("Спросите об условиях карты, кредита или рассрочки"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Ищу в условиях и считаю…"):
                data = to_data(agent.ask(question, use_graph=use_graph))
        except llm.LLMUnavailable as e:
            st.error(f"Не удалось обратиться к локальной модели. {e}\n\n"
                     f"Проверьте: `ollama serve` запущен, модели скачаны (`ollama pull {llm.DEFAULT_MODEL}`, "
                     f"`ollama pull {llm.EMBED_MODEL}`).")
        else:
            show_response(data)
            st.session_state.messages.append({"role": "assistant", "data": data})
