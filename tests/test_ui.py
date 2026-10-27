"""Этап 5: интерфейс через streamlit.testing (без браузера), LLM подменён."""

import json
import unittest
from unittest import mock

from streamlit.testing.v1 import AppTest

from rag import llm


def fake_chat(messages, schema=None, **kw):
    q = messages[-1]["content"]
    if schema:
        if "100 000" in q:
            return json.dumps({"route": "tool", "tool": "loan_calculator", "sum": 100000, "rate": 18, "months": 24,
                               "product_hints": []})
        return json.dumps({"route": "rag", "tool": "none", "product_hints": []})
    return "**Подводные камни**\nПропуск платежа отменяет льготный период.\n\n**Итог**\nПлатите вовремя.\nИСПОЛЬЗОВАНО: f1,f4"


class UITest(unittest.TestCase):
    def run_app(self, question, chat):
        # Эмбеддинг запроса подменяем: тест не должен зависеть от того, запущен ли Ollama и чем собран data/chroma.
        # Если коллекция построена bge-m3, без подмены поиск идёт в настоящий Ollama и тест падает без него.
        fake_embed = lambda texts, *a, **k: [[0.0] * 1024 for _ in texts]  # noqa: E731
        with mock.patch.object(llm, "chat", side_effect=chat), mock.patch.object(llm, "available", return_value=True), \
                mock.patch.object(llm, "embed", side_effect=fake_embed):
            at = AppTest.from_file("../ui/app.py", default_timeout=120).run()
            at.chat_input[0].set_value(question).run()
        return at

    def test_answer_sources_and_graph(self):
        at = self.run_app("Что отменяет льготный период по карте «Полярная 120»?", fake_chat)
        text = " ".join(m.value for m in at.markdown)
        self.assertIn("Подводные камни", text)
        self.assertIn("**Источники**", text)
        self.assertIn("sever_credit.md", text)
        self.assertIn("Цепочка в графе", [e.label for e in at.expander])
        self.assertTrue(any("учебный проект" in i.value.lower() for i in at.info))

    def test_calculation_expander(self):
        at = self.run_app("Кредит 100 000 ₽ на 2 года под 18%, какая переплата?", fake_chat)
        self.assertIn("Показать расчёт: loan_calculator", [e.label for e in at.expander])
        self.assertIn("19817.84", json.dumps([j.value for j in at.json]))
        at.session_state  # агент и MCP-подпроцесс остаются в кэше ресурса

    def test_ollama_error_is_explained(self):
        def down(*a, **k):
            raise llm.LLMUnavailable("Ollama недоступен по http://localhost:11434")
        at = self.run_app("Что отменяет льготный период по карте «Полярная 120»?", down)
        self.assertTrue(at.error)
        self.assertIn("ollama serve", at.error[0].value)

    def test_refusal_without_llm(self):
        at = self.run_app("Какая погода завтра?", fake_chat)
        self.assertIn("отвечаю только", " ".join(m.value for m in at.markdown))


if __name__ == "__main__":
    unittest.main()
