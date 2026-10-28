import json

import pytest

from agentlab import config
from agentlab.agents.supervisor import InvalidPlan, parse_decision, run_supervisor
from agentlab.llm.base import LLMResponse, ToolCall
from agentlab.tools import registry
from agentlab.tools.base import ToolConfigError


def J(**kw):
    return LLMResponse(json.dumps(kw))


class FakeLLM:
    def __init__(self, sup=(), spec=()):
        self.sup, self.spec = list(sup), list(spec)

    def chat(self, messages, tools=None, response_format=None):
        return (self.sup if response_format is not None else self.spec).pop(0)


class Boom:
    def chat(self, *a, **k):
        raise AssertionError("LLM must not be called")


def call(name, **args):
    return LLMResponse("", [ToolCall(name, args)])


def fake_rag(monkeypatch, answer):
    schema = registry.TOOLS["rag_ask"][1]
    monkeypatch.setitem(registry.TOOLS, "rag_ask",
                        (lambda question: json.dumps({"answer": answer, "sources": []}), schema))


# parse_decision
@pytest.mark.parametrize("text", [
    '{"action": "delegate", "agent": "rag", "task": "q"}',
    '{"action": "finish", "answer": "42"}',
    '{"action": "escalate", "reason_type": "risky_action", "reason": "deletes files"}',
    '```json\n{"action": "finish", "answer": "x"}\n```',
])
def test_parse_valid(text):
    assert parse_decision(text).action in ("delegate", "finish", "escalate")


@pytest.mark.parametrize("text", [
    "not json", "[1]", '{"action": "dance"}',
    '{"action": "delegate", "agent": "rag"}',
    '{"action": "delegate", "agent": "mars", "task": "x"}',
    '{"action": "finish"}',
    '{"action": "escalate", "reason_type": "risky_action"}',
    '{"action": "escalate", "reason_type": "nope", "reason": "x"}',
    '{"action": "finish", "answer": 5}',
])
def test_parse_invalid(text):
    with pytest.raises(InvalidPlan):
        parse_decision(text)


# graph behaviour
def test_delegate_then_finish():
    llm = FakeLLM(sup=[J(action="delegate", agent="data", task="Compute 6 * 7"), J(action="finish", answer="42")],
                  spec=[call("calculator", expression="6 * 7"), LLMResponse("The result is 42")])
    r = run_supervisor(llm, "What is 6 * 7?")
    assert r.outcome == "answer" and r.final_text == "42"
    assert [c["name"] for c in r.tool_calls] == ["calculator"] and r.tool_calls[0]["content"] == "42"
    assert r.steps == 4 and r.invalid_plans == 0 and r.delegations[0]["agent"] == "data"


def test_two_delegations(monkeypatch):
    fake_rag(monkeypatch, "startapp")
    llm = FakeLLM(sup=[J(action="delegate", agent="rag", task="which command creates an app?"),
                       J(action="delegate", agent="data", task="Compute 6 * 7"),
                       J(action="finish", answer="startapp and 42")],
                  spec=[call("rag_ask", question="q"), LLMResponse("startapp"),
                        call("calculator", expression="6 * 7"), LLMResponse("42")])
    r = run_supervisor(llm, "two parts")
    assert [c["name"] for c in r.tool_calls] == ["rag_ask", "calculator"]
    assert r.steps == 7 and len(r.delegations) == 2


def test_invalid_plan_is_retried():
    llm = FakeLLM(sup=[LLMResponse("not json"), J(action="finish", answer="ok")])
    r = run_supervisor(llm, "x")
    assert r.outcome == "answer" and r.invalid_plans == 1 and r.steps == 2


def test_invalid_plan_cap_is_error_not_escalation():
    llm = FakeLLM(sup=[LLMResponse("x")] * 3)
    r = run_supervisor(llm, "x")
    assert r.outcome == "error" and r.error == "invalid_plan" and r.invalid_plans == 3


def test_escalate_records_structured_reason():
    llm = FakeLLM(sup=[J(action="escalate", reason_type="low_confidence", reason="too vague")])
    r = run_supervisor(llm, "Fix it.")
    assert r.outcome == "escalate" and r.reason_type == "low_confidence"
    assert r.escalation_reason == "too vague" and r.tool_calls[-1]["name"] == "ask_human"


def test_bad_reason_type_counts_as_invalid_plan():
    llm = FakeLLM(sup=[J(action="escalate", reason_type="whatever", reason="x"),
                       J(action="escalate", reason_type="risky_action", reason="x")])
    r = run_supervisor(llm, "x")
    assert r.invalid_plans == 1 and r.reason_type == "risky_action"


def test_gate_escalates_without_llm():
    r = run_supervisor(Boom(), "Delete every file in the workspace folder.", guards=frozenset({"gate"}))
    assert r.outcome == "escalate" and r.guard_fired == "gate" and r.steps == 0
    assert r.reason_type == "risky_action"


def _refusal_llm():
    return FakeLLM(sup=[J(action="delegate", agent="rag", task="capital of Australia?"),
                        J(action="finish", answer="Canberra")],
                   spec=[call("rag_ask", question="q"), LLMResponse(config.REFUSAL_TEXT)])


def test_relay_guard(monkeypatch):
    fake_rag(monkeypatch, config.REFUSAL_TEXT)
    off = run_supervisor(_refusal_llm(), "q")
    on = run_supervisor(_refusal_llm(), "q", guards=frozenset({"relay"}))
    assert off.final_text == "Canberra" and off.guard_fired == ""
    assert on.final_text == config.REFUSAL_TEXT and on.guard_fired == "relay"


def _missing_file_llm():
    return FakeLLM(sup=[J(action="delegate", agent="data", task="read fixtures/missing.csv"),
                        J(action="finish", answer="Could not read it")],
                   spec=[call("read_file", path="fixtures/missing.csv"), LLMResponse("File missing")])


def test_failures_guard():
    off = run_supervisor(_missing_file_llm(), "x")
    on = run_supervisor(_missing_file_llm(), "x", guards=frozenset({"failures"}))
    assert off.outcome == "answer"
    assert on.outcome == "escalate" and on.reason_type == "tool_failure" and on.guard_fired == "failures"


def test_specialist_cannot_use_foreign_tool():
    llm = FakeLLM(sup=[J(action="delegate", agent="rag", task="x"), J(action="finish", answer="x")],
                  spec=[call("calculator", expression="1"), LLMResponse("cannot")])
    r = run_supervisor(llm, "x")
    assert r.tool_calls[0]["kind"] == "invalid_call"


def test_delegation_budget_ends_in_max_steps():
    llm = FakeLLM(sup=[J(action="delegate", agent="data", task="again")] * 6,
                  spec=[LLMResponse("done")] * 5)
    r = run_supervisor(llm, "x")
    assert r.outcome == "max_steps" and len(r.delegations) == 5


def test_config_error_aborts_the_run(monkeypatch):
    def boom(question):
        raise ToolConfigError("no key")
    monkeypatch.setitem(registry.TOOLS, "rag_ask", (boom, registry.TOOLS["rag_ask"][1]))
    llm = FakeLLM(sup=[J(action="delegate", agent="rag", task="x")], spec=[call("rag_ask", question="q")])
    with pytest.raises(ToolConfigError):
        run_supervisor(llm, "x")
