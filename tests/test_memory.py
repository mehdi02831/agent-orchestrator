import psycopg
import pytest

from agentlab import config
from agentlab.agents.single import RunResult
from agentlab.agents.supervisor import SupervisorResult
from agentlab.memory.records import run_record
from agentlab.memory.store import PostgresRunStore

TEST_DB = "agentlab_test"


def _sup():
    return SupervisorResult(
        outcome="answer", final_text="42", steps=7, latency_s=18.3, max_prompt_tokens=651, invalid_plans=1,
        tool_calls=[{"name": "rag_ask", "arguments": {"question": "q"}, "kind": "ok", "content": "a"},
                    {"name": "calculator", "arguments": {"expression": "6 * 7"}, "kind": "ok", "content": "42"}],
        delegations=[{"agent": "rag", "task": "t1", "result": "r1"}, {"agent": "data", "task": "t2", "result": "r2"}])


# pure tests (no database)
def test_record_from_single_result():
    rec = run_record("single", "hi", RunResult(outcome="escalate", escalation_reason="vague", steps=1))
    assert rec["status"] == "escalate" and rec["escalation_reason"] == "vague"
    assert rec["reason_type"] == "" and rec["invalid_plans"] == 0 and rec["delegations"] == []


def test_record_from_supervisor_result():
    rec = run_record("supervisor", "q", _sup(), {"guards": ["gate"]})
    assert rec["invalid_plans"] == 1 and rec["steps"] == 7 and rec["config"] == {"guards": ["gate"]}
    assert [t["name"] for t in rec["tool_calls"]] == ["rag_ask", "calculator"]


def test_record_accepts_plain_dict():
    rec = run_record("x", "p", {"outcome": "answer", "final_text": "ok"})
    assert rec["status"] == "answer" and rec["steps"] == 0 and rec["tool_calls"] == []


# integration tests (throwaway database; skipped if Postgres is unreachable)
@pytest.fixture(scope="module")
def store():
    try:
        admin = psycopg.connect(**config.postgres_conn("postgres"), autocommit=True)
    except Exception as e:
        pytest.skip(f"Postgres not reachable: {type(e).__name__}")
    with admin:
        admin.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
        admin.execute(f"CREATE DATABASE {TEST_DB}")
    s = PostgresRunStore(**config.postgres_conn(TEST_DB))
    s.init_schema()
    yield s
    with psycopg.connect(**config.postgres_conn("postgres"), autocommit=True) as admin:
        admin.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")


def test_roundtrip_supervisor(store):
    rid = store.save_run(run_record("supervisor", "q", _sup(), {"guards": ["gate"]}))
    got = store.get_run(rid)
    assert got["agent"] == "supervisor" and got["status"] == "answer" and got["final_text"] == "42"
    assert got["steps"] == 7 and got["invalid_plans"] == 1 and got["config"] == {"guards": ["gate"]}
    assert got["tool_calls"][1]["arguments"] == {"expression": "6 * 7"}


def test_tool_call_and_delegation_order_preserved(store):
    got = store.get_run(store.save_run(run_record("supervisor", "q", _sup())))
    assert [t["name"] for t in got["tool_calls"]] == ["rag_ask", "calculator"]
    assert [d["agent"] for d in got["delegations"]] == ["rag", "data"]


def test_second_store_instance_reads_saved_run(store):
    rid = store.save_run(run_record("supervisor", "q", _sup()))
    other = PostgresRunStore(**config.postgres_conn(TEST_DB))  # new object, new connections
    assert other.get_run(rid)["final_text"] == "42"
    assert other.get_run(10**9) is None


def test_list_runs_newest_first(store):
    store.save_run(run_record("single", "first", RunResult(outcome="answer")))
    last = store.save_run(run_record("single", "second", RunResult(outcome="answer")))
    ids = [r["id"] for r in store.list_runs(2)]
    assert ids[0] == last and ids == sorted(ids, reverse=True)


def test_unicode_kept_and_nul_stripped(store):
    prompt = "caf\u00e9 26\u00b0C"
    rid = store.save_run(run_record("single", prompt, RunResult(outcome="answer", final_text="a\x00b \u00b0F")))
    got = store.get_run(rid)
    assert got["prompt"] == prompt and got["final_text"] == "ab \u00b0F"


def test_init_schema_is_idempotent(store):
    store.init_schema()
    store.init_schema()


def test_delete_run_cascades(store):
    rid = store.save_run(run_record("supervisor", "q", _sup()))
    assert store.delete_run(rid) is True
    assert store.get_run(rid) is None
    assert store.delete_run(rid) is False
    with psycopg.connect(**config.postgres_conn(TEST_DB)) as c:
        n = c.execute("SELECT count(*) FROM run_tool_calls WHERE run_id = %s", (rid,)).fetchone()[0]
    assert n == 0


def test_set_status(store):
    rid = store.save_run(run_record("supervisor", "q", _sup()))
    assert store.set_status(rid, "approved") is True
    assert store.get_run(rid)["status"] == "approved"
    assert store.set_status(10**9, "x") is False
