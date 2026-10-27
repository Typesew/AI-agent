"""MCP-сервер на FastMCP: 4 детерминированных инструмента, транспорт stdio.

    python -m mcp_server.server        # обычно его запускает agent.client как подпроцесс
"""

from typing import Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError as MCPToolError

from mcp_server import tools

mcp = FastMCP("bank-translator-tools", instructions="Расчёты по кредитам и картам. Числа считает код.")


def _run(fn, *args):
    try:
        return fn(*args)
    except tools.ToolError as e:
        raise MCPToolError(str(e)) from e


@mcp.tool
def loan_calculator(sum: float, rate: float, months: int, type: Literal["annuity", "diff"] = "annuity",
                    fee: float = 0.0) -> dict:
    """Ежемесячный платёж, общая сумма и переплата. rate — годовая ставка в %, 0 для рассрочки."""
    return _run(tools.loan_calculator, sum, rate, months, type, fee)


@mcp.tool
def grace_period_calc(purchase_date: str, statement_day: int = 0, grace_days: int = 0,
                      mode: Literal["from_purchase", "from_month_start", "base_plus_extension"] = "from_purchase") -> dict:
    """Последний день, когда можно вернуть долг без процентов (период считается включительно)."""
    return _run(tools.grace_period_calc, purchase_date, statement_day, grace_days, mode)


@mcp.tool
def compare_products(ids: list[str]) -> dict:
    """Таблица сравнения 2–5 продуктов по обслуживанию, снятию наличных, льготному периоду и кэшбэку."""
    return _run(tools.compare_products, ids)


@mcp.tool
def save_summary(title: str, text: str) -> dict:
    """Сохранить разбор в файл data/summaries/<title>.md."""
    return _run(tools.save_summary, title, text)


if __name__ == "__main__":
    mcp.run(show_banner=False)
