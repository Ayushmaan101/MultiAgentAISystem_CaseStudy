import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import httpx
from agno.agent import Agent
from agno.models.groq import Groq
from asteval import Interpreter

import config

_http_client = httpx.Client(verify=False)

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
    model=Groq(
        id=config.GROQ_MODEL,
        api_key=config.GROQ_API_KEY,
        http_client=_http_client,
    ),
    tools=[safe_calculate],
    instructions=[
        "You are a mathematical computation assistant.",
        "Always call safe_calculate for every math query — never compute yourself.",
        "After the tool returns, clearly state the expression and its numeric result.",
    ],
    markdown=True,
)
