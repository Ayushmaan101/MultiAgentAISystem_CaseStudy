import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agno.agent import Agent
from asteval import Interpreter

import config
from llm_client import get_model

_aeval = Interpreter()


def safe_calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression using asteval (never Python eval)."""
    _aeval.error.clear()
    result = _aeval(expression)
    if _aeval.error:
        error_msgs = "; ".join(str(e.get_error()) for e in _aeval.error)
        return f"Expression: {expression}\nError: {error_msgs}"
    return f"Expression: {expression}\nResult: {result}"


calculator_agent = Agent(
    name="Calculator Agent",
    model=get_model(),
    tools=[safe_calculate],
    instructions=[
        "You are a mathematical computation assistant.",
        "You MUST call the safe_calculate tool for every math query. Do NOT compute yourself.",
        "Step 1: Call safe_calculate with the mathematical expression.",
        "Step 2: Copy the tool result verbatim into your response.",
        "Step 3: State the expression and its numeric result clearly.",
    ],
    markdown=True,
)
