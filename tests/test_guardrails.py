import json
from pathlib import Path

import pytest

from agentlab import config
from agentlab.agents.guardrails import is_risky, refusal_relay, unrecovered_failure
from agentlab.agents.single import run_single
from agentlab.llm.base import LLMResponse, ToolCall
from agentlab.tools import registry


@pytest.mark.parametrize("p", [
    "Delete every file in the workspace folder.",
    "Overwrite fixtures/prices.csv with empty content.",
    "Write a summary of the Django ORM docs to workspace/out/summary.md.",
    "Please remove notes.txt",
    "Save the report into fixtures/report.md",
])
def test_gate_positive(p):
    assert is_risky(p)


@pytest.mark.parametrize("p", [
    "How do I delete an object in Django?",
    "How do I create a new Django app inside a project?",
    "How do I save a file upload in Django?",
    "Read fixtures/prices.csv and give the total cost, summing qty * unit_price.",
    "Fetch https://docs.invalid.example/data.json and summarize it.",
    "Fix it.",
])
def test_gate_negative(p):
    assert not is_risky(p)


def test_gate_matches_dev_split_only():
    # Reads the dev split only. The frozen test split is deliberately not touched here.
    path = Path(__file__).resolve().parents[1] / "eval" / "tasks.jsonl"
    tasks = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    for t in tasks:
        if t["split"] == "dev":
            assert is_risky(t["prompt"]) == (t["expect"].get("reason") == "risky_action"), t["id"]


def _c(name, kind, content=""):
    return {"name": name, "kind": kind, "content": content}


REFUSAL_JSON = json.dumps({"answer": config.REFUSAL_TEXT, "sources": []})


def test_relay_true_when_only_refusals():
    assert refusal_relay([_c("rag_ask", "ok", REFUSAL_JSON)])


def test_relay_false_with_real_answer():
    assert not refusal_relay([_c("rag_ask", "ok", json.dumps({"answer": "startapp", "sources": ["x"]}))])


def test_relay_false_when_other_tool_succeeded():
    assert not refusal_relay([_c("rag_ask", "ok", REFUSAL_JSON), _c("calculator", "ok", "42")])


def test_relay_false_without_rag():
    assert not refusal_relay([])


def test_relay_false_on_truncated_json():
    assert not refusal_relay([_c("rag_ask", "ok", '{"answer": "x')])


def test_failure_unrecovered():
    assert unrecovered_failure([_c("web_fetch", "tool_error"), _c("web_fetch", "tool_error")])


def test_failure_recovered():
    assert not unrecovered_failure([_c("web_fetch", "tool_error"), _c("web_fetch", "ok", "x")])


def test_failure_ignores_invalid_calls_and_ask_human():
    assert not unrecovered_failure([_c("calculator", "invalid_call"), _c("ask_human", "ok")])


def test_failure_other_tool_ok_still_counts():
    assert unrecovered_failure([_c("read_file", "tool_error"), _c("calculator", "ok", "1")])


class Fake:
    def __init__(self, *script):
        self.script = list(script)

    def chat(self, messages, tools=None):
        return self.script.pop(0)


class Boom:
    def chat(self, *a, **k):
        raise AssertionError("LLM must not be called")


def _fake_rag(monkeypatch, answer):
    schema = registry.TOOLS["rag_ask"][1]
    monkeypatch.setitem(registry.TOOLS, "rag_ask",
                        (lambda question: json.dumps({"answer": answer, "sources": []}), schema))


def _rag_call():
    return LLMResponse("", [ToolCall("rag_ask", {"question": "q"})])


def test_gate_escalates_without_llm():
    r = run_single(Boom(), "Delete every file in the workspace folder.", guards=frozenset({"gate"}))
    assert r.outcome == "escalate" and r.guard_fired == "gate" and r.steps == 0


def test_gate_off_by_default():
    r = run_single(Fake(LLMResponse("ok")), "Delete every file in the workspace folder.")
    assert r.outcome == "answer" and r.guard_fired == ""


def test_relay_overrides_final_text(monkeypatch):
    _fake_rag(monkeypatch, config.REFUSAL_TEXT)
    off = run_single(Fake(_rag_call(), LLMResponse("Canberra")), "q")
    on = run_single(Fake(_rag_call(), LLMResponse("Canberra")), "q", guards=frozenset({"relay"}))
    assert off.final_text == "Canberra" and off.guard_fired == ""
    assert on.final_text == config.REFUSAL_TEXT and on.guard_fired == "relay"


def test_relay_not_counted_when_text_already_exact(monkeypatch):
    _fake_rag(monkeypatch, config.REFUSAL_TEXT)
    r = run_single(Fake(_rag_call(), LLMResponse(config.REFUSAL_TEXT)), "q", guards=frozenset({"relay"}))
    assert r.guard_fired == "" and r.final_text == config.REFUSAL_TEXT


def test_failures_guard_turns_giving_up_into_escalation():
    def script():
        return Fake(LLMResponse("", [ToolCall("read_file", {"path": "fixtures/missing.csv"})]),
                    LLMResponse("I could not find the file."))
    off = run_single(script(), "x")
    on = run_single(script(), "x", guards=frozenset({"failures"}))
    assert off.outcome == "answer"
    assert on.outcome == "escalate" and on.guard_fired == "failures"


def test_unsupported_refusal_without_rag():
    from agentlab.agents.guardrails import unsupported_refusal
    assert unsupported_refusal(config.REFUSAL_TEXT, [])
    assert unsupported_refusal(config.REFUSAL_TEXT, [_c("web_fetch", "tool_error")])


def test_supported_refusal_and_other_answers():
    from agentlab.agents.guardrails import unsupported_refusal
    assert not unsupported_refusal(config.REFUSAL_TEXT, [_c("rag_ask", "ok", REFUSAL_JSON)])
    assert not unsupported_refusal("Canberra", [])
    real = json.dumps({"answer": "startapp", "sources": ["x"]})
    assert unsupported_refusal(config.REFUSAL_TEXT, [_c("rag_ask", "ok", real)])
