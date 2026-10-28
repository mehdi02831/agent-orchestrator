from dataclasses import dataclass
from .. import config
from .base import ToolError
from .calculator import calculator
from .files import read_file
from .rag import rag_ask
from .web import web_fetch


def _schema(name, desc, param, param_desc):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object",
                       "properties": {param: {"type": "string", "description": param_desc}},
                       "required": [param]}}}


TOOLS = {
    "rag_ask": (rag_ask, _schema(
        "rag_ask", "Answer a question about the Django documentation, with sources. "
                   "Returns the exact text 'I don't know based on the documentation.' if the docs do not cover it.",
        "question", "The question to ask the Django docs")),
    "calculator": (calculator, _schema(
        "calculator", "Evaluate an arithmetic expression using + - * / ** % and parentheses. "
                      "Write percentages as decimals, for example 15% of 200 is 0.15 * 200.",
        "expression", "The arithmetic expression")),
    "read_file": (read_file, _schema(
        "read_file", "Read a text file from the workspace folder, for example fixtures/prices.csv.",
        "path", "Path relative to the workspace folder")),
    "web_fetch": (web_fetch, _schema(
        "web_fetch", "Fetch an https page from an allow-listed host.",
        "url", "The https URL")),
}


@dataclass
class ToolResult:
    ok: bool
    content: str
    kind: str  # "ok" | "invalid_call" | "tool_error"


def tool_schemas(names=None):
    return [spec[1] for n, spec in TOOLS.items() if names is None or n in names]


def execute(name, arguments) -> ToolResult:
    spec = TOOLS.get(name)
    if spec is None:
        return ToolResult(False, f"unknown tool '{name}'", "invalid_call")
    func, schema = spec
    params = schema["function"]["parameters"]
    if not isinstance(arguments, dict):
        return ToolResult(False, "arguments must be an object", "invalid_call")
    missing = [k for k in params["required"] if k not in arguments]
    extra = [k for k in arguments if k not in params["properties"]]
    wrong = [k for k in arguments if k in params["properties"] and not isinstance(arguments[k], str)]
    if missing or extra or wrong:
        return ToolResult(False, f"invalid arguments (missing={missing}, unexpected={extra}, not_string={wrong})",
                          "invalid_call")
    try:
        out = func(**arguments)  # ToolConfigError is NOT caught: it aborts the run
    except ToolError as e:
        return ToolResult(False, str(e), "tool_error")
    if len(out) > config.MAX_TOOL_OUTPUT_CHARS:
        out = out[:config.MAX_TOOL_OUTPUT_CHARS] + "\n[truncated]"
    return ToolResult(True, out, "ok")
