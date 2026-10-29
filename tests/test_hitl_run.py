import json
from pathlib import Path

import pytest

from agentlab import config
from agentlab.agents.hitl_run import extract_target_path, resolve, run_with_review
from agentlab.llm.base import LLMResponse, ToolCall
from agentlab.tools import registry

WRITE = "Write a summary of the Django ORM docs to workspace/out/summary.md."
G = frozenset({"gate"})


class Seq:
    def __init__(self, *responses):
        self.q, self.calls = list(responses), 0

    def chat(self, messages, tools=None, response_format=None):
        self.calls += 1
        return self.q.pop(0)


class Boom:
    def chat(self, *a, **k):
        raise AssertionError("LLM must not be called")


class MemRuns:
    def __init__(self):
        self.rows, self.n = {}, 0

    def save_run(self, rec):
        self.n += 1
        self.rows[self.n] = dict(rec)
        return self.n

    def set_status(self, rid, status):
        if rid not in self.rows:
            return False
        self.rows[rid]["status"] = status
        return True


class MemStore:
    def __init__(self):
        self.rows, self.n = {}, 0

    def create(self, rec):
        self.n += 1
        self.rows[self.n] = {"id": self.n, "status": "pending", "decision": "", "edited": None,
                             "result": "", "last_error": "", "note": "", **rec}
        return self.n

    def get(self, i):
        return self.rows.get(i)

    def claim(self, i):
        r = self.rows.get(i)
        if r is None or r["status"] != "pending":
            return None
        r["status"] = "resolving"
        return dict(r)

    def release(self, i, error):
        self.rows[i].update(status="pending", last_error=error)

    def finish(self, i, status, decision, edited, result, note):
        self.rows[i].update(status=status, decision=decision, edited=edited, result=result, note=note)


@pytest.fixture(autouse=True)
def out_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WRITE_DIR", tmp_path)
    return tmp_path


def fake_rag(monkeypatch, answer):
    schema = registry.TOOLS["rag_ask"][1]
    monkeypatch.setitem(registry.TOOLS, "rag_ask",
                        (lambda question: json.dumps({"answer": answer, "sources": []}), schema))


def rag_call():
    return LLMResponse("", [ToolCall("rag_ask", {"question": "q"})])


def good_seq(draft=None):
    return Seq(rag_call(), LLMResponse("The ORM maps models to tables."),
               LLMResponse(json.dumps(draft or {"content": "# ORM\nMaps models to tables."})))


def queued(monkeypatch):
    fake_rag(monkeypatch, "The ORM maps models to tables.")
    runs, pend = MemRuns(), MemStore()
    return runs, pend, run_with_review(good_seq(), WRITE, runs, pend, guards=G)


def test_write_request_is_queued_then_approved(monkeypatch, out_dir):
    runs, pend, out = queued(monkeypatch)
    row = pend.get(out.pending_id)
    assert out.proposal == "queued" and out.result.guard_fired == "gate"
    assert row["run_id"] == out.run_id and row["proposed"]["path"] == "workspace/out/summary.md"
    assert runs.rows[out.run_id]["status"] == "escalate"
    assert not (out_dir / "summary.md").exists()
    status, _ = resolve(runs, pend, out.pending_id, "approve")
    assert status == "approved"
    assert (out_dir / "summary.md").read_text(encoding="utf-8") == "# ORM\nMaps models to tables."
    assert runs.rows[out.run_id]["status"] == "approved"


def test_reject_updates_run_and_writes_nothing(monkeypatch, out_dir):
    runs, pend, out = queued(monkeypatch)
    assert resolve(runs, pend, out.pending_id, "reject")[0] == "rejected"
    assert runs.rows[out.run_id]["status"] == "rejected" and not list(out_dir.iterdir())


def test_edit_writes_human_text(monkeypatch, out_dir):
    runs, pend, out = queued(monkeypatch)
    assert resolve(runs, pend, out.pending_id, "edit", edits={"content": "human"})[0] == "approved"
    assert (out_dir / "summary.md").read_text(encoding="utf-8") == "human"
    assert runs.rows[out.run_id]["status"] == "edited"


def test_failed_write_leaves_run_status(monkeypatch, out_dir):
    runs, pend, out = queued(monkeypatch)
    (out_dir / "summary.md").write_text("old", encoding="utf-8")
    assert resolve(runs, pend, out.pending_id, "approve")[0] == "pending"
    assert runs.rows[out.run_id]["status"] == "escalate"
    assert (out_dir / "summary.md").read_text(encoding="utf-8") == "old"


def test_delete_request_gets_no_proposal():
    runs, pend = MemRuns(), MemStore()
    out = run_with_review(Boom(), "Delete every file in the workspace folder.", runs, pend, guards=G)
    assert out.result.outcome == "escalate" and out.proposal == "not_write_request"
    assert out.pending_id is None and pend.rows == {} and len(runs.rows) == 1


def test_invalid_target_makes_no_llm_call():
    runs, pend = MemRuns(), MemStore()
    out = run_with_review(Boom(), "Write the summary to ../escape.md", runs, pend, guards=G)
    assert out.proposal == "invalid_target" and pend.rows == {}


def test_not_risky_run_is_only_saved():
    runs, pend = MemRuns(), MemStore()
    seq = Seq(LLMResponse(json.dumps({"action": "finish", "answer": "42"})))
    out = run_with_review(seq, "What is 6 * 7?", runs, pend, guards=G)
    assert out.proposal == "not_risky" and out.result.outcome == "answer" and seq.calls == 1
    assert pend.rows == {} and runs.rows[out.run_id]["status"] == "answer"


def test_refusal_source_makes_no_draft(monkeypatch):
    fake_rag(monkeypatch, config.REFUSAL_TEXT)
    pend = MemStore()
    seq = Seq(rag_call(), LLMResponse(config.REFUSAL_TEXT))
    out = run_with_review(seq, WRITE, MemRuns(), pend, guards=G)
    assert out.proposal == "refusal" and seq.calls == 2 and pend.rows == {}


def test_no_rag_call_means_no_source():
    pend = MemStore()
    out = run_with_review(Seq(LLMResponse("From memory.")), WRITE, MemRuns(), pend, guards=G)
    assert out.proposal == "no_source" and pend.rows == {}


def test_bad_draft_json(monkeypatch):
    fake_rag(monkeypatch, "x")
    pend = MemStore()
    seq = Seq(rag_call(), LLMResponse("text"), LLMResponse("not json"))
    assert run_with_review(seq, WRITE, MemRuns(), pend, guards=G).proposal == "bad_draft"
    assert pend.rows == {}


def test_empty_draft_content(monkeypatch):
    fake_rag(monkeypatch, "x")
    seq = Seq(rag_call(), LLMResponse("text"), LLMResponse(json.dumps({"content": "   "})))
    assert run_with_review(seq, WRITE, MemRuns(), MemStore(), guards=G).proposal == "bad_draft"


def test_oversize_draft(monkeypatch):
    fake_rag(monkeypatch, "x")
    monkeypatch.setattr(config, "MAX_WRITE_CHARS", 5)
    seq = Seq(rag_call(), LLMResponse("text"), LLMResponse(json.dumps({"content": "x" * 10})))
    pend = MemStore()
    assert run_with_review(seq, WRITE, MemRuns(), pend, guards=G).proposal == "bad_draft"
    assert pend.rows == {}


def test_model_supplied_path_is_ignored(monkeypatch):
    fake_rag(monkeypatch, "x")
    pend = MemStore()
    seq = good_seq({"content": "x", "path": "../../evil.md"})
    out = run_with_review(seq, WRITE, MemRuns(), pend, guards=G)
    assert out.proposal == "queued" and pend.get(out.pending_id)["proposed"]["path"] == "workspace/out/summary.md"


@pytest.mark.parametrize("prompt,expected", [
    ("Write a summary to workspace/out/summary.md.", "workspace/out/summary.md"),
    ("Save the report into fixtures/report.md", "fixtures/report.md"),
    ("Export the totals to the file out/totals.csv, please", "out/totals.csv"),
    ("Write hello", None),
])
def test_extract_target_path(prompt, expected):
    assert extract_target_path(prompt) == expected


def test_frozen_runner_never_imports_review_code():
    src = (Path(__file__).resolve().parents[1] / "eval" / "run_eval.py").read_text(encoding="utf-8")
    assert "hitl_run" not in src


from agentlab.agents.hitl_run import source_question, topic_from_prompt


@pytest.mark.parametrize("prompt,path,topic", [
    ("Write a summary of the Django ORM docs to workspace/out/summary.md.", "workspace/out/summary.md", "Django ORM"),
    ("Save a short summary of Django migrations to workspace/out/migrations.md", "workspace/out/migrations.md",
     "Django migrations"),
    ("Write what the startapp command does into workspace/out/startapp.txt", "workspace/out/startapp.txt",
     "what the startapp command does"),
    ("Export an explanation of select_related to workspace/out/orm.md", "workspace/out/orm.md", "select_related"),
])
def test_topic_from_prompt(prompt, path, topic):
    assert topic_from_prompt(prompt, path) == topic


@pytest.mark.parametrize("topic,question", [
    ("Django ORM", "Explain Django ORM."),
    ("select_related", "Explain select_related in Django."),
])
def test_source_question(topic, question):
    assert source_question(topic) == question


def test_no_topic_makes_no_llm_call():
    pend = MemStore()
    out = run_with_review(Boom(), "Write it to workspace/out/a.md", MemRuns(), pend, guards=G)
    assert out.proposal == "no_topic" and pend.rows == {}


def test_specialist_receives_the_topic_question(monkeypatch):
    fake_rag(monkeypatch, "The ORM maps models to tables.")
    seen = []

    class Rec(Seq):
        def chat(self, messages, tools=None, response_format=None):
            seen.append(messages)
            return super().chat(messages, tools, response_format)

    run_with_review(Rec(rag_call(), LLMResponse("text"), LLMResponse(json.dumps({"content": "x"}))),
                    WRITE, MemRuns(), MemStore(), guards=G)
    assert "Explain Django ORM." in seen[0][1]["content"] and "workspace" not in seen[0][1]["content"]
