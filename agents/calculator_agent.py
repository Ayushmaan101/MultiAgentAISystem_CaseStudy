import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from asteval import Interpreter

_aeval = Interpreter()


def safe_calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression using asteval (never Python eval)."""
    _aeval.error.clear()
    result = _aeval(expression)
    if _aeval.error:
        error_msgs = "; ".join(str(e.get_error()) for e in _aeval.error)
        return f"Expression: {expression}\nError: {error_msgs}"
    return f"Expression: {expression}\nResult: {result}"
