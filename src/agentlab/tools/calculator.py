import ast
import operator
from .base import ToolError

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv}
_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _ev(n):
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
        return n.value
    if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
        left, right = _ev(n.left), _ev(n.right)
        if isinstance(n.op, ast.Pow) and abs(right) > 1000:
            raise ToolError("exponent too large")
        return _BIN[type(n.op)](left, right)
    if isinstance(n, ast.UnaryOp) and type(n.op) in _UN:
        return _UN[type(n.op)](_ev(n.operand))
    raise ToolError(f"unsupported element: {type(n).__name__}")


def calculator(expression: str) -> str:
    if len(expression) > 200:
        raise ToolError("expression too long")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as e:
        raise ToolError(f"invalid expression: {e.msg}")
    try:
        value = _ev(tree.body)
    except ZeroDivisionError:
        raise ToolError("division by zero")
    except OverflowError:
        raise ToolError("number too large")
    if isinstance(value, float):
        value = round(value, 10)
    return str(value)
