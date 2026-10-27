"""MCP-клиент: сервер стартует подпроцессом (stdio), сессия живёт, пока жив клиент.

Streamlit и CLI синхронные, а MCP SDK асинхронный, поэтому сессия работает
в отдельном потоке со своим event loop, а call() ждёт результат синхронно.
"""

import asyncio
import json
import os
import sys
import threading
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parent.parent
SERVER_LOG = ROOT / "data" / "mcp_server.log"


class ToolCallError(RuntimeError):
    """Инструмент вернул ошибку (например, неверные аргументы)."""


class MCPClient:
    def __init__(self, command: list[str] | None = None, timeout: float = 60):
        cmd = command or [sys.executable, "-m", "mcp_server.server"]
        # SDK по умолчанию передаёт подпроцессу только PATH/HOME и т.п.; нам нужны
        # OLLAMA_URL, BANK_SUMMARY_DIR, BANK_GRAPH_DB.
        self.params = StdioServerParameters(command=cmd[0], args=cmd[1:], cwd=str(ROOT), env=dict(os.environ))
        self.timeout = timeout
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._session: ClientSession | None = None
        self._thread = threading.Thread(target=lambda: asyncio.run(self._main()), daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise ToolCallError("MCP-сервер не запустился за отведённое время")
        if self._error:
            raise ToolCallError(f"MCP-сервер не запустился: {self._error}") from self._error

    async def _main(self):
        try:
            SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(SERVER_LOG, "a", encoding="utf-8") as log:  # stderr сервера — в файл, не в терминал
                async with stdio_client(self.params, errlog=log) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        self._session, self._loop, self._stop = session, asyncio.get_running_loop(), asyncio.Event()
                        self._ready.set()
                        await self._stop.wait()
        except BaseException as e:  # noqa: BLE001 — пробрасываем в основной поток
            self._error = e
            self._ready.set()

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(self.timeout)

    def list_tools(self) -> list[str]:
        return [t.name for t in self._run(self._session.list_tools()).tools]

    def call(self, name: str, arguments: dict) -> dict:
        res = self._run(self._session.call_tool(name, arguments))
        text = "".join(getattr(c, "text", "") for c in res.content)
        if res.is_error:
            raise ToolCallError(text or f"{name}: ошибка инструмента")
        if res.structured_content is not None:
            return res.structured_content
        return json.loads(text)

    def close(self):
        if self._loop and self._stop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=10)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
