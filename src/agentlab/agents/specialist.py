from dataclasses import dataclass, field

from .. import config
from ..llm.base import LLMProvider
from ..tools.registry import ToolResult, execute, tool_schemas

MAX_SPECIALIST_STEPS = 4

SPECIALISTS = {
    "rag": {
        "tools": ["rag_ask"],
        "prompt": ("You answer questions about the Django documentation using the rag_ask tool. "
                   "Always call rag_ask first. If it answers \"" + config.REFUSAL_TEXT + "\", reply with exactly "
                   "that sentence and nothing else. Otherwise reply with a short answer based only on the tool result."),
    },
    "data": {
        "tools": ["calculator", "read_file"],
        "prompt": ("You do arithmetic with the calculator tool and read files with the read_file tool "
                   "(paths are relative to the workspace folder, for example fixtures/prices.csv). "
                   "Never do arithmetic in your head. Reply with a short plain-text result."),
    },
    "web": {
        "tools": ["web_fetch"],
        "prompt": ("You fetch https pages with the web_fetch tool (allow-listed hosts only). "
                   "Reply with a short plain-text summary of what the tool returned. "
                   "If the tool returns an error, say so plainly."),
    },
}


@dataclass
class SpecialistOutput:
    text: str = ""
    tool_calls: list = field(default_factory=list)
    llm_calls: int = 0
    max_prompt_tokens: int = 0
    finished: bool = False


def run_specialist(provider: LLMProvider, name: str, task: str,
                   max_steps: int = MAX_SPECIALIST_STEPS) -> SpecialistOutput:
    """Small ReAct loop restricted to this specialist's tools. ToolConfigError is not caught."""
    spec = SPECIALISTS[name]
    allowed = set(spec["tools"])
    messages = [{"role": "system", "content": spec["prompt"]}, {"role": "user", "content": task}]
    schemas = tool_schemas(allowed)
    out = SpecialistOutput()
    for _ in range(max_steps):
        resp = provider.chat(messages, schemas)
        out.llm_calls += 1
        out.max_prompt_tokens = max(out.max_prompt_tokens, resp.raw.get("prompt_eval_count", 0))
        if not resp.tool_calls:
            out.text, out.finished = resp.content.strip(), True
            return out
        messages.append({"role": "assistant", "content": resp.content,
                         "tool_calls": [{"function": {"name": c.name, "arguments": c.arguments}}
                                        for c in resp.tool_calls]})
        for c in resp.tool_calls:
            if c.name not in allowed:
                r = ToolResult(False, f"tool '{c.name}' is not available to this agent", "invalid_call")
            else:
                r = execute(c.name, c.arguments)
            out.tool_calls.append({"name": c.name, "arguments": c.arguments,
                                   "kind": r.kind, "content": r.content[:300]})
            messages.append({"role": "tool", "tool_name": c.name, "content": r.content})
    out.text = "The specialist did not finish within its step limit."
    return out
