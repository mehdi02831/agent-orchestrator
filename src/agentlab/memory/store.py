from abc import ABC, abstractmethod

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .. import config

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS runs (
        id BIGSERIAL PRIMARY KEY,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        agent TEXT NOT NULL,
        prompt TEXT NOT NULL,
        config JSONB NOT NULL DEFAULT '{}',
        status TEXT NOT NULL,
        final_text TEXT NOT NULL DEFAULT '',
        escalation_reason TEXT NOT NULL DEFAULT '',
        reason_type TEXT NOT NULL DEFAULT '',
        guard_fired TEXT NOT NULL DEFAULT '',
        error TEXT NOT NULL DEFAULT '',
        steps INT NOT NULL DEFAULT 0,
        invalid_plans INT NOT NULL DEFAULT 0,
        latency_s DOUBLE PRECISION NOT NULL DEFAULT 0,
        max_prompt_tokens INT NOT NULL DEFAULT 0)""",
    """CREATE TABLE IF NOT EXISTS run_tool_calls (
        run_id BIGINT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        seq INT NOT NULL,
        name TEXT NOT NULL,
        arguments JSONB NOT NULL DEFAULT '{}',
        kind TEXT NOT NULL DEFAULT '',
        content TEXT NOT NULL DEFAULT '',
        PRIMARY KEY (run_id, seq))""",
    """CREATE TABLE IF NOT EXISTS run_delegations (
        run_id BIGINT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        seq INT NOT NULL,
        agent TEXT NOT NULL,
        task TEXT NOT NULL DEFAULT '',
        result TEXT NOT NULL DEFAULT '',
        PRIMARY KEY (run_id, seq))""",
]


class RunStore(ABC):
    @abstractmethod
    def init_schema(self) -> None: ...

    @abstractmethod
    def save_run(self, record: dict) -> int: ...

    @abstractmethod
    def get_run(self, run_id: int) -> dict | None: ...

    @abstractmethod
    def list_runs(self, limit: int = 20) -> list[dict]: ...

    @abstractmethod
    def delete_run(self, run_id: int) -> bool: ...


class PostgresRunStore(RunStore):
    def __init__(self, **conn_kwargs):
        self.conn_kwargs = conn_kwargs

    @classmethod
    def from_config(cls):
        return cls(**config.postgres_conn())

    def _connect(self):
        return psycopg.connect(**self.conn_kwargs)  # the with-block commits on success and closes

    def init_schema(self):
        with self._connect() as c:
            for stmt in SCHEMA:
                c.execute(stmt)

    def save_run(self, rec):
        with self._connect() as c:
            cur = c.execute(
                "INSERT INTO runs (agent, prompt, config, status, final_text, escalation_reason, reason_type,"
                " guard_fired, error, steps, invalid_plans, latency_s, max_prompt_tokens)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (rec["agent"], rec["prompt"], Jsonb(rec["config"]), rec["status"], rec["final_text"],
                 rec["escalation_reason"], rec["reason_type"], rec["guard_fired"], rec["error"],
                 rec["steps"], rec["invalid_plans"], rec["latency_s"], rec["max_prompt_tokens"]))
            rid = cur.fetchone()[0]
            with c.cursor() as cur2:
                cur2.executemany(
                    "INSERT INTO run_tool_calls (run_id, seq, name, arguments, kind, content) VALUES (%s,%s,%s,%s,%s,%s)",
                    [(rid, i, t.get("name", ""), Jsonb(t.get("arguments", {})), t.get("kind", ""), t.get("content", ""))
                     for i, t in enumerate(rec["tool_calls"])])
                cur2.executemany(
                    "INSERT INTO run_delegations (run_id, seq, agent, task, result) VALUES (%s,%s,%s,%s,%s)",
                    [(rid, i, d.get("agent", ""), d.get("task", ""), d.get("result", ""))
                     for i, d in enumerate(rec["delegations"])])
        return rid

    def get_run(self, run_id):
        with self._connect() as c:
            c.row_factory = dict_row
            run = c.execute("SELECT * FROM runs WHERE id = %s", (run_id,)).fetchone()
            if run is None:
                return None
            run["tool_calls"] = c.execute(
                "SELECT name, arguments, kind, content FROM run_tool_calls WHERE run_id = %s ORDER BY seq",
                (run_id,)).fetchall()
            run["delegations"] = c.execute(
                "SELECT agent, task, result FROM run_delegations WHERE run_id = %s ORDER BY seq",
                (run_id,)).fetchall()
        return run

    def list_runs(self, limit=20):
        with self._connect() as c:
            c.row_factory = dict_row
            return c.execute(
                "SELECT id, created_at, agent, status, prompt FROM runs ORDER BY id DESC LIMIT %s",
                (limit,)).fetchall()

    def delete_run(self, run_id):
        with self._connect() as c:
            cur = c.execute("DELETE FROM runs WHERE id = %s", (run_id,))
            return cur.rowcount > 0
