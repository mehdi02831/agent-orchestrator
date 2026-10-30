import pytest

from agentlab.jobs.runner import execute_task
from agentlab.llm.base import LLMResponse
from agentlab.tools.base import ToolConfigError


class MemRuns:
    def __init__(self):
        self.rows, self.n = {}, 0

    def save_run(self, rec):
        self.n += 1
        self.rows[self.n] = dict(rec)
        return self.n

    def set_status(self, rid, status):
        self.rows[rid]["status"] = status
        return True


class Finish:
    def chat(self, messages, tools=None, response_format=None):
        return LLMResponse('{"action": "finish", "answer": "42"}')


class Raises:
    def __init__(self, exc):
        self.exc = exc

    def chat(self, *a, **k):
        raise self.exc


class FailingRuns(MemRuns):
    def save_run(self, rec):
        raise RuntimeError("db down")


def test_answer_run_is_saved_and_summarised():
    runs = MemRuns()
    out = execute_task("What is 6 * 7?", Finish(), runs, object())
    assert out == {"run_id": 1, "outcome": "answer", "proposal": "not_risky", "pending_id": None, "final": "42"}
    assert runs.rows[1]["status"] == "answer" and runs.rows[1]["final_text"] == "42"


def test_gate_escalation_without_llm():
    out = execute_task("Delete every file in the workspace folder.", Raises(AssertionError("no LLM")),
                       MemRuns(), object())
    assert out["outcome"] == "escalate" and out["proposal"] == "not_write_request"


def test_config_error_is_saved_and_reraised():
    runs = MemRuns()
    with pytest.raises(ToolConfigError):
        execute_task("What is 6 * 7?", Raises(ToolConfigError("no key")), runs, object())
    assert runs.rows[1]["status"] == "error" and runs.rows[1]["error"] == "ToolConfigError"


def test_unexpected_error_is_saved_and_reraised():
    runs = MemRuns()
    with pytest.raises(RuntimeError):
        execute_task("What is 6 * 7?", Raises(RuntimeError("boom")), runs, object())
    assert runs.rows[1]["error"] == "RuntimeError"


def test_save_failure_does_not_mask_the_original_error():
    with pytest.raises(ToolConfigError):
        execute_task("What is 6 * 7?", Raises(ToolConfigError("no key")), FailingRuns(), object())
