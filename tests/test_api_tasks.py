from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from agentlab import config
from agentlab.api import app as api_module
from agentlab.hitl import propose

client = TestClient(api_module.app)
H = {"X-API-Key": "secret-key"}


class MemRuns:
    def __init__(self):
        self.rows = {}

    def get_run(self, rid):
        return self.rows.get(rid)

    def set_status(self, rid, status):
        if rid not in self.rows:
            return False
        self.rows[rid]["status"] = status
        return True


class MemPending:
    def __init__(self):
        self.rows, self.n = {}, 0

    def create(self, rec):
        self.n += 1
        self.rows[self.n] = {"id": self.n, "status": "pending", "decision": "", "edited": None,
                             "result": "", "last_error": "", "note": "", **rec}
        return self.n

    def get(self, i):
        return self.rows.get(i)

    def list_open(self):
        return [r for r in self.rows.values() if r["status"] == "pending"]

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
def key(monkeypatch):
    monkeypatch.setattr(config, "ORCH_API_KEY", "secret-key")


@pytest.fixture
def stores(monkeypatch, tmp_path):
    runs, pend = MemRuns(), MemPending()
    runs.rows[1] = {"id": 1, "status": "escalate"}
    monkeypatch.setattr(api_module, "get_run_store", lambda: runs)
    monkeypatch.setattr(api_module, "get_pending_store", lambda: pend)
    monkeypatch.setattr(config, "WRITE_DIR", tmp_path)
    return runs, pend, tmp_path


def fake_result(state, result=None):
    return lambda task_id, app=None: SimpleNamespace(state=state, result=result,
                                                     successful=lambda: state == "SUCCESS")


def queue(pend, path="a.md", content="draft"):
    return propose(pend, path, content, prompt="p", reason="r", run_id=1)


def test_every_new_endpoint_requires_the_key():
    assert client.post("/tasks", json={"prompt": "hi"}).status_code == 401
    assert client.get("/tasks/x").status_code == 401
    assert client.get("/runs/1").status_code == 401
    assert client.get("/pending").status_code == 401
    assert client.post("/pending/1/decide", json={"decision": "approve"}).status_code == 401


def test_submit_enqueues_the_prompt(monkeypatch):
    sent = []
    monkeypatch.setattr(api_module.jobs, "run_task",
                        SimpleNamespace(delay=lambda p: sent.append(p) or SimpleNamespace(id="t1")))
    r = client.post("/tasks", json={"prompt": "What is 6 * 7?"}, headers=H)
    assert r.status_code == 200 and r.json() == {"task_id": "t1"} and sent == ["What is 6 * 7?"]


def test_submit_validation():
    assert client.post("/tasks", json={"prompt": ""}, headers=H).status_code == 422
    assert client.post("/tasks", json={"prompt": "   "}, headers=H).status_code == 422
    assert client.post("/tasks", json={"prompt": "x" * 2001}, headers=H).status_code == 422
    assert client.post("/tasks", json={}, headers=H).status_code == 422


def test_status_success_includes_the_run(monkeypatch, stores):
    runs, _, _ = stores
    runs.rows[7] = {"id": 7, "status": "answer"}
    monkeypatch.setattr(api_module, "AsyncResult", fake_result("SUCCESS", {"run_id": 7, "outcome": "answer"}))
    body = client.get("/tasks/t1", headers=H).json()
    assert body["state"] == "SUCCESS" and body["result"]["run_id"] == 7 and body["run"]["status"] == "answer"


def test_status_pending_has_no_result(monkeypatch):
    monkeypatch.setattr(api_module, "AsyncResult", fake_result("PENDING"))
    body = client.get("/tasks/t1", headers=H).json()
    assert body["state"] == "PENDING" and body["result"] is None and body["run"] is None


def test_status_failure_reports_the_error(monkeypatch):
    monkeypatch.setattr(api_module, "AsyncResult", fake_result("FAILURE", RuntimeError("boom")))
    body = client.get("/tasks/t1", headers=H).json()
    assert body["state"] == "FAILURE" and body["error"] == "boom"


def test_run_endpoint(stores):
    assert client.get("/runs/999", headers=H).status_code == 404
    assert client.get("/runs/1", headers=H).json()["status"] == "escalate"


def test_pending_lists_only_open_items(stores):
    _, pend, _ = stores
    a, b = queue(pend, "a.md"), queue(pend, "b.md")
    client.post(f"/pending/{a}/decide", json={"decision": "reject"}, headers=H)
    items = client.get("/pending", headers=H).json()["pending"]
    assert [i["id"] for i in items] == [b] and items[0]["proposed"]["content"] == "draft"


def test_decide_approve_writes_and_updates_the_run(stores):
    runs, pend, out_dir = stores
    pid = queue(pend)
    assert not (out_dir / "a.md").exists()
    r = client.post(f"/pending/{pid}/decide", json={"decision": "approve", "note": "ok"}, headers=H)
    assert r.status_code == 200 and r.json()["status"] == "approved"
    assert (out_dir / "a.md").read_text(encoding="utf-8") == "draft"
    assert runs.rows[1]["status"] == "approved"


def test_decide_reject_writes_nothing(stores):
    runs, pend, out_dir = stores
    pid = queue(pend)
    r = client.post(f"/pending/{pid}/decide", json={"decision": "reject"}, headers=H)
    assert r.json()["status"] == "rejected" and not list(out_dir.iterdir()) and runs.rows[1]["status"] == "rejected"


def test_decide_edit_writes_the_human_text(stores):
    runs, pend, out_dir = stores
    pid = queue(pend)
    r = client.post(f"/pending/{pid}/decide", json={"decision": "edit", "edits": {"content": "human"}}, headers=H)
    assert r.status_code == 200 and (out_dir / "a.md").read_text(encoding="utf-8") == "human"
    assert runs.rows[1]["status"] == "edited"


def test_second_decision_is_a_conflict(stores):
    _, pend, _ = stores
    pid = queue(pend)
    assert client.post(f"/pending/{pid}/decide", json={"decision": "approve"}, headers=H).status_code == 200
    assert client.post(f"/pending/{pid}/decide", json={"decision": "approve"}, headers=H).status_code == 409


def test_invalid_decision_is_rejected_by_validation(stores):
    _, pend, _ = stores
    pid = queue(pend)
    assert client.post(f"/pending/{pid}/decide", json={"decision": "maybe"}, headers=H).status_code == 422


def test_bad_edits_are_a_client_error(stores):
    _, pend, _ = stores
    pid = queue(pend)
    assert client.post(f"/pending/{pid}/decide", json={"decision": "edit"}, headers=H).status_code == 400
    bad = client.post(f"/pending/{pid}/decide", json={"decision": "edit", "edits": {"colour": "red"}}, headers=H)
    assert bad.status_code == 400 and pend.get(pid)["status"] == "pending"


def test_failed_write_stays_pending(stores):
    _, pend, out_dir = stores
    (out_dir / "a.md").write_text("old", encoding="utf-8")
    pid = queue(pend)
    r = client.post(f"/pending/{pid}/decide", json={"decision": "approve"}, headers=H)
    assert r.status_code == 200 and r.json()["status"] == "pending" and "overwrite" in r.json()["message"]
    assert (out_dir / "a.md").read_text(encoding="utf-8") == "old"
