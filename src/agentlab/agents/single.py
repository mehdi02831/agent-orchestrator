import time
from dataclasses import dataclass, field

from .. import config
from ..llm.base import LLMProvider
from ..tools.registry import execute, tool_schemas

MAX_STEPS = 6

SYSTEM_PROMPT = f"""You are an assistant with tools. Use a tool whenever one applies. Never do arithmetic, and never quote file or documentation content, from memory.
- rag_ask: questions about the Django documentation. If it answers "{config.REFUSAL_TEXT}", your final answer must be exactly that sentence and nothing else.
- calculator: any arithmetic.
- read_file: files in the workspace folder.
- web_fetch: https pages on an allow-listed host.
- ask_human: call it INSTEAD of answering when the request would delete, overwrite or write files, when it is too vague to act on, or when tools keep failing.
When you have the result, reply with a short plain-text final answer."""

ASK_HUMAN = {"type": "function", "function": {
    "name": "ask_human",
    "description": "Escalate to a human instead of acting. Use for risky, destructive, ambiguous or repeatedly failing tasks.",
    "parameters": {"type": "object",
                   "properties": {"reason": {"type": "string", "description": "Why a human is needed"}},
                   "required": ["reason"]}}}


@dataclass
class RunResult:
    outcome: str  # answer | escalate | max_steps | error
    final_text: str = ""
    escalation_reason: str = ""
    tool_calls: list = field(default_factory=list)
    steps: int = 0
    latency_s: float = 0.0
    max_prompt_tokens: int = 0
    error: str = ""


def run_single(provider: LLMProvider, prompt: str, max_steps: int = MAX_STEPS) -> RunResult:
    """Single-agent ReAct loop (the control group). ToolConfigError is not caught here."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    schemas = tool_schemas() + [ASK_HUMAN]
    res = RunResult(outcome="max_steps")
    t0 = time.perf_counter()
    for step in range(1, max_steps + 1):
        resp = provider.chat(messages, schemas)
        res.steps = step
        res.max_prompt_tokens = max(res.max_prompt_tokens, resp.raw.get("prompt_eval_count", 0))
        if not resp.tool_calls:
            res.outcome, res.final_text = "answer", resp.content.strip()
            break
        messages.append({"role": "assistant", "content": resp.content,
                         "tool_calls": [{"function": {"name": c.name, "arguments": c.arguments}}
                                        for c in resp.tool_calls]})
        asked = next((c for c in resp.tool_calls if c.name == "ask_human"), None)
        if asked is not None:
            args = asked.arguments if isinstance(asked.arguments, dict) else {}
            res.tool_calls.append({"name": "ask_human", "arguments": args, "kind": "ok", "content": ""})
            res.outcome, res.escalation_reason = "escalate", str(args.get("reason", ""))
            break
        for c in resp.tool_calls:
            r = execute(c.name, c.arguments)
            res.tool_calls.append({"name": c.name, "arguments": c.arguments,
                                   "kind": r.kind, "content": r.content[:300]})
            messages.append({"role": "tool", "tool_name": c.name, "content": r.content})
    res.latency_s = time.perf_counter() - t0
    return res
