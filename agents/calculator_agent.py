import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from asteval import Interpreter

import config

_aeval = Interpreter()


def safe_calculate(expression: str) -> dict:
    """Safely evaluate a mathematical expression using asteval (never Python eval)."""
    _aeval.error.clear()
    result = _aeval(expression)
    if _aeval.error:
        error_msgs = "; ".join(str(e.get_error()) for e in _aeval.error)
        return {"expression": expression, "error": error_msgs}
    return {"expression": expression, "result": result}


calculator_agent = Agent(
    name="Calculator Agent",
    model=OpenAILike(
        id=config.LLM_MODEL,
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
    ),
    tools=[safe_calculate],
    instructions=[
        "You are a mathematical computation assistant.",
        "Always use the safe_calculate tool for any arithmetic, algebraic, or numerical computation.",
        "Never compute math yourself — always delegate to the tool.",
        "Show the expression and its result clearly in your response.",
    ],
    markdown=True,
)
