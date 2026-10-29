import psycopg
import pytest

from agentlab import config
from agentlab.hitl import HitlError, decide, propose
from agentlab.memory.pending import PostgresPendingStore
from agentlab.memory.records import run_record
from agentlab.memory.store import PostgresRunStore
from agentlab.tools.base import ToolError

TEST_DB = "agentlab_test_hitl"


class MemStore:
    def __init__(self):
        self.rows, self.n = {}, 0

    def init_schema(self):
        pass

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
def out_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WRITE_DIR", tmp_path)
    return tmp_path


@pytest.fixture(scope="module")
def pg_store():
    try:
        admin = psycopg.connect(**config.postgres_conn("postgres"), autocommit=True)
    except Exception as e:
        pytest.skip(f"Postgres not reachable: {type(e).__name__}")
    with admin:
        admin.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
        admin.execute(f"CREATE DATABASE {TEST_DB}")
    s = PostgresPendingStore(**config.postgres_conn(TEST_DB))
    s.init_schema()
    yield s
    with psycopg.connect(**config.postgres_conn("postgres"), autocommit=True) as admin:
        admin.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")


@pytest.fixture(params=["mem", "pg"])
def store(request):
    if request.param == "mem":
        return MemStore()
    return request.getfixturevalue("pg_store")


def _new(store, path="a.md", content="draft"):
    return propose(store, path, content, prompt="write it", reason="writes a file")


def test_approve_writes_file(store, out_dir):
    pid = _new(store)
    assert not (out_dir / "a.md").exists()
    status, msg = decide(store, pid, "approve", note="ok")
    row = store.get(pid)
    assert status == "approved" and (out_dir / "a.md").read_text(encoding="utf-8") == "draft"
    assert row["status"] == "approved" and row["decision"] == "approve" and "wrote" in row["result"]


def test_reject_writes_nothing(store, out_dir):
    pid = _new(store)
    assert decide(store, pid, "reject")[0] == "rejected"
    assert not list(out_dir.iterdir()) and store.get(pid)["status"] == "rejected"


def test_edit_changes_what_is_written(store, out_dir):
    pid = _new(store)
    assert decide(store, pid, "edit", edits={"content": "human text", "path": "b.md"})[0] == "approved"
    assert (out_dir / "b.md").read_text(encoding="utf-8") == "human text" and not (out_dir / "a.md").exists()
    row = store.get(pid)
    assert row["decision"] == "edit" and row["edited"]["path"] == "b.md"


def test_second_decision_refused(store, out_dir):
    pid = _new(store)
    decide(store, pid, "approve")
    with pytest.raises(HitlError, match="already resolved"):
        decide(store, pid, "approve")
    with pytest.raises(HitlError):
        decide(store, pid, "reject")


def test_unknown_id(store):
    with pytest.raises(HitlError, match="not found"):
        decide(store, 10**9, "approve")


def test_failed_write_returns_to_pending_and_can_retry(store, out_dir):
    pid = _new(store)
    status, msg = decide(store, pid, "edit", edits={"path": "bad.exe"})
    row = store.get(pid)
    assert status == "pending" and row["status"] == "pending" and "not allowed" in row["last_error"]
    assert decide(store, pid, "edit", edits={"path": "ok.md"})[0] == "approved"
    assert (out_dir / "ok.md").exists()


def test_existing_file_needs_explicit_overwrite(store, out_dir):
    (out_dir / "a.md").write_text("old", encoding="utf-8")
    pid = _new(store)
    status, msg = decide(store, pid, "approve")
    assert status == "pending" and "overwrite" in msg
    assert (out_dir / "a.md").read_text(encoding="utf-8") == "old"
    assert decide(store, pid, "edit", edits={"overwrite": True})[0] == "approved"
    assert (out_dir / "a.md").read_text(encoding="utf-8") == "draft"


def test_invalid_decisions_leave_row_pending(store):
    pid = _new(store)
    for args in [("maybe", None), ("edit", None), ("edit", {"colour": "red"}), ("approve", {"content": "x"})]:
        with pytest.raises(HitlError):
            decide(store, pid, args[0], edits=args[1])
    assert store.get(pid)["status"] == "pending"


def test_list_open_excludes_resolved(store, out_dir):
    a, b = _new(store, "a.md"), _new(store, "b.md")
    decide(store, a, "reject")
    ids = [r["id"] for r in store.list_open()]
    assert b in ids and a not in ids


def test_claim_has_a_single_winner(store):
    pid = _new(store)
    assert store.claim(pid) is not None
    assert store.claim(pid) is None


def test_propose_refuses_bad_path_and_queues_nothing():
    s = MemStore()
    with pytest.raises(ToolError):
        propose(s, "../escape.md", "x")
    assert s.rows == {}


def test_pg_second_instance_reads_pending(pg_store):
    pid = propose(pg_store, "x.md", "hi", prompt="p", reason="r")
    other = PostgresPendingStore(**config.postgres_conn(TEST_DB))
    row = other.get(pid)
    assert row["status"] == "pending" and row["proposed"]["content"] == "hi" and row["reason"] == "r"
    assert pid in [r["id"] for r in other.list_open()]


def test_pg_run_deletion_keeps_pending(pg_store):
    runs = PostgresRunStore(**config.postgres_conn(TEST_DB))
    runs.init_schema()
    rid = runs.save_run(run_record("supervisor", "q", {"outcome": "escalate"}))
    pid = propose(pg_store, "y.md", "hi", run_id=rid)
    assert pg_store.get(pid)["run_id"] == rid
    runs.delete_run(rid)
    assert pg_store.get(pid)["run_id"] is None
