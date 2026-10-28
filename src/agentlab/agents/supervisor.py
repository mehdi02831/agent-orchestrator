import json
import re
import time
from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .. import config
from ..llm.base import LLMProvider
from .guardrails import is_risky, refusal_relay, unrecovered_failure
from .single import RunResult
from .specialist import run_specialist

AGENTS = ("rag", "data", "web")
REASON_TYPES = ("risky_action", "tool_failure", "low_confidence")
MAX_TURNS = 5          # delegations allowed; the next supervisor call must finish or escalate
MAX_PLAN_ATTEMPTS = 3  # first try plus two retries after an invalid reply

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["delegate", "finish", "escalate"]},
        "agent": {"type": "string", "enum": list(AGENTS) + [""]},
        "task": {"type": "string"},
        "answer": {"type": "string"},
        "reason_type": {"type": "string", "enum": list(REASON_TYPES) + [""]},
        "reason": {"type": "string"},
    },
    "required": ["action"],
}

SUPERVISOR_PROMPT = """You are a supervisor. You never use tools yourself. Reply with ONE JSON object and nothing else.
Specialists:
- rag: answers questions about the Django documentation. It returns the exact sentence "__REFUSAL__" when the docs do not cover the question.
- data: arithmetic (calculator) and reading files in the workspace folder (read_file).
- web: fetches https pages from allow-listed hosts.
Actions:
{"action": "delegate", "agent": "rag|data|web", "task": "<self-contained instruction>"}
{"action": "finish", "answer": "<final short plain-text answer>"}
{"action": "escalate", "reason_type": "risky_action|tool_failure|low_confidence", "reason": "<why a human is needed>"}
Rules:
1. Any general-knowledge or Django question MUST be delegated to rag first. Never answer such a question from memory.
2. If rag returned "__REFUSAL__", finish with exactly that sentence. Add nothing: no apology, no other source, no guess.
3. Arithmetic and file reading go to data. Split multi-part requests: one delegation per part, then finish with a combined answer.
4. Escalate instead of finishing when: the request is to delete, overwrite, move or write a file (no specialist can); the request is too vague to act on; or a specialist reported a tool error twice.
5. Delegated tasks must be self-contained: include file paths and numbers."""
SUPERVISOR_PROMPT = SUPERVISOR_PROMPT.replace("__REFUSAL__", config.REFUSAL_TEXT)


class InvalidPlan(Exception):
    pass


@dataclass
class Decision:
    action: str
    agent: str = ""
    task: str = ""
    answer: str = ""
    reason_type: str = ""
    reason: str = ""


def parse_decision(text: str) -> Decision:
    s = (text or "").strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.S)
    if m:
        s = m.group(1)
    try:
        data = json.loads(s)
    except ValueError:
        raise InvalidPlan("not valid JSON")
    if not isinstance(data, dict):
        raise InvalidPlan("JSON must be an object")

    def _s(key):
        v = data.get(key, "")
        if v is None:
            v = ""
        if not isinstance(v, str):
            raise InvalidPlan(f"'{key}' must be a string")
        return v.strip()

    d = Decision(_s("action"), _s("agent"), _s("task"), _s("answer"), _s("reason_type"), _s("reason"))
    if d.action == "delegate":
        if d.agent not in AGENTS:
            raise InvalidPlan(f"agent must be one of {list(AGENTS)}")
        if not d.task:
            raise InvalidPlan("delegate needs a non-empty task")
    elif d.action == "finish":
        if not d.answer:
            raise InvalidPlan("finish needs a non-empty answer")
    elif d.action == "escalate":
        if d.reason_type not in REASON_TYPES:
            raise InvalidPlan(f"reason_type must be one of {list(REASON_TYPES)}")
        if not d.reason:
            raise InvalidPlan("escalate needs a non-empty reason")
    else:
        raise InvalidPlan("action must be delegate, finish or escalate")
    return d


@dataclass
class SupervisorResult(RunResult):
    invalid_plans: int = 0
    reason_type: str = ""
    delegations: list = field(default_factory=list)


class State(TypedDict):
    prompt: str
    history: list
    tool_calls: list
    steps: int
    turns: int
    invalid_plans: int
    max_prompt_tokens: int
    next_agent: str
    next_task: str
    outcome: str
    final_text: str
    reason: str
    reason_type: str
    error: str
    guard_fired: str


def _messages(prompt, history, feedback):
    msgs = [{"role": "system", "content": SUPERVISOR_PROMPT}, {"role": "user", "content": prompt}]
    for h in history:
        msgs.append({"role": "assistant",
                     "content": json.dumps({"action": "delegate", "agent": h["agent"], "task": h["task"]})})
        msgs.append({"role": "user", "content": f"Result from {h['agent']}: {h['result']}"})
    msgs.extend(feedback)
    return msgs


def build_graph(provider: LLMProvider, guards=frozenset()):
    def supervisor(state: State) -> dict:
        feedback = []
        steps, invalid, mpt = state["steps"], state["invalid_plans"], state["max_prompt_tokens"]
        for _ in range(MAX_PLAN_ATTEMPTS):
            resp = provider.chat(_messages(state["prompt"], state["history"], feedback), None,
                                 response_format=DECISION_SCHEMA)
            steps += 1
            mpt = max(mpt, resp.raw.get("prompt_eval_count", 0))
            try:
                d = parse_decision(resp.content)
            except InvalidPlan as e:
                invalid += 1
                feedback += [{"role": "assistant", "content": resp.content},
                             {"role": "user", "content": f"Invalid reply: {e}. Reply with one valid JSON object."}]
                continue
            base = {"steps": steps, "invalid_plans": invalid, "max_prompt_tokens": mpt}
            if d.action == "delegate":
                if state["turns"] >= MAX_TURNS:
                    return {**base, "outcome": "max_steps"}
                return {**base, "next_agent": d.agent, "next_task": d.task, "turns": state["turns"] + 1}
            if d.action == "finish":
                return {**base, "outcome": "answer", "final_text": d.answer}
            return {**base, "outcome": "escalate", "reason": d.reason, "reason_type": d.reason_type}
        return {"steps": steps, "invalid_plans": invalid, "max_prompt_tokens": mpt,
                "outcome": "error", "error": "invalid_plan"}

    def specialist_node(name):
        def node(state: State) -> dict:
            out = run_specialist(provider, name, state["next_task"])
            result = out.text
            errs = [f"{c['name']}: {c['content']}" for c in out.tool_calls if c["kind"] != "ok"]
            if errs:
                result += " [tool errors: " + "; ".join(errs) + "]"
            return {"history": state["history"] + [{"agent": name, "task": state["next_task"], "result": result}],
                    "tool_calls": state["tool_calls"] + out.tool_calls,
                    "steps": state["steps"] + out.llm_calls,
                    "max_prompt_tokens": max(state["max_prompt_tokens"], out.max_prompt_tokens),
                    "next_agent": ""}
        return node

    def finalize(state: State) -> dict:
        if state["outcome"] == "answer":
            if ("relay" in guards and refusal_relay(state["tool_calls"])
                    and state["final_text"].strip() != config.REFUSAL_TEXT):
                return {"final_text": config.REFUSAL_TEXT, "guard_fired": "relay"}
            if "failures" in guards and unrecovered_failure(state["tool_calls"]):
                return {"outcome": "escalate", "reason": "guard: tool failure not recovered",
                        "reason_type": "tool_failure", "guard_fired": "failures"}
        return {"guard_fired": state["guard_fired"]}

    def route(state: State) -> str:
        return "finalize" if state["outcome"] else state["next_agent"]

    g = StateGraph(State)
    g.add_node("supervisor", supervisor)
    for name in AGENTS:
        g.add_node(name, specialist_node(name))
        g.add_edge(name, "supervisor")
    g.add_node("finalize", finalize)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", route, {**{n: n for n in AGENTS}, "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


def run_supervisor(provider: LLMProvider, prompt: str, guards=frozenset()) -> SupervisorResult:
    """Supervisor + specialists. ToolConfigError and LLM/network errors propagate to the caller."""
    t0 = time.perf_counter()
    if "gate" in guards and is_risky(prompt):
        reason = "guard: request to delete, overwrite or write files"
        return SupervisorResult(
            outcome="escalate", escalation_reason=reason, reason_type="risky_action", guard_fired="gate",
            tool_calls=[{"name": "ask_human", "arguments": {"reason": reason}, "kind": "ok", "content": ""}],
            latency_s=time.perf_counter() - t0)
    graph = build_graph(provider, guards)
    st = graph.invoke({"prompt": prompt, "history": [], "tool_calls": [], "steps": 0, "turns": 0,
                       "invalid_plans": 0, "max_prompt_tokens": 0, "next_agent": "", "next_task": "",
                       "outcome": "", "final_text": "", "reason": "", "reason_type": "", "error": "",
                       "guard_fired": ""}, {"recursion_limit": 40})
    calls = list(st["tool_calls"])
    if st["outcome"] == "escalate":
        calls.append({"name": "ask_human", "arguments": {"reason": st["reason"]}, "kind": "ok", "content": ""})
    return SupervisorResult(
        outcome=st["outcome"], final_text=st["final_text"], escalation_reason=st["reason"],
        tool_calls=calls, steps=st["steps"], latency_s=time.perf_counter() - t0,
        max_prompt_tokens=st["max_prompt_tokens"], error=st["error"], guard_fired=st["guard_fired"],
        invalid_plans=st["invalid_plans"], reason_type=st["reason_type"],
        delegations=[{"agent": h["agent"], "task": h["task"]} for h in st["history"]])
