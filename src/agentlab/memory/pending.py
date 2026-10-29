from abc import ABC, abstractmethod

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .. import config
from .records import _clean
from .store import SCHEMA as RUN_SCHEMA

PENDING_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS pending_actions (
        id BIGSERIAL PRIMARY KEY,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        decided_at TIMESTAMPTZ,
        run_id BIGINT REFERENCES runs(id) ON DELETE SET NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        prompt TEXT NOT NULL DEFAULT '',
        reason TEXT NOT NULL DEFAULT '',
        reason_type TEXT NOT NULL DEFAULT '',
        proposed JSONB NOT NULL,
        state JSONB NOT NULL DEFAULT '{}',
        decision TEXT NOT NULL DEFAULT '',
        edited JSONB,
        result TEXT NOT NULL DEFAULT '',
        last_error TEXT NOT NULL DEFAULT '',
        note TEXT NOT NULL DEFAULT '')""",
]


class PendingStore(ABC):
    @abstractmethod
    def init_schema(self) -> None: ...

    @abstractmethod
    def create(self, rec: dict) -> int: ...

    @abstractmethod
    def get(self, pending_id: int) -> dict | None: ...

    @abstractmethod
    def list_open(self) -> list[dict]: ...

    @abstractmethod
    def claim(self, pending_id: int) -> dict | None:
        """Atomically move pending -> resolving. Returns the row, or None if not pending."""

    @abstractmethod
    def release(self, pending_id: int, error: str) -> None: ...

    @abstractmethod
    def finish(self, pending_id: int, status: str, decision: str, edited, result: str, note: str) -> None: ...


class PostgresPendingStore(PendingStore):
    def __init__(self, **conn_kwargs):
        self.conn_kwargs = conn_kwargs

    @classmethod
    def from_config(cls):
        return cls(**config.postgres_conn())

    def _connect(self):
        return psycopg.connect(**self.conn_kwargs)

    def init_schema(self):
        with self._connect() as c:
            for stmt in RUN_SCHEMA + PENDING_SCHEMA:
                c.execute(stmt)

    def create(self, rec):
        rec = _clean(rec)
        with self._connect() as c:
            cur = c.execute(
                "INSERT INTO pending_actions (run_id, prompt, reason, reason_type, proposed, state)"
                " VALUES (%s,%s,%s,%s,%s,%s) RETURNING id",
                (rec.get("run_id"), rec.get("prompt", ""), rec.get("reason", ""), rec.get("reason_type", ""),
                 Jsonb(rec["proposed"]), Jsonb(rec.get("state", {}))))
            return cur.fetchone()[0]

    def get(self, pending_id):
        with self._connect() as c:
            c.row_factory = dict_row
            return c.execute("SELECT * FROM pending_actions WHERE id = %s", (pending_id,)).fetchone()

    def list_open(self):
        with self._connect() as c:
            c.row_factory = dict_row
            return c.execute("SELECT * FROM pending_actions WHERE status = 'pending' ORDER BY id").fetchall()

    def claim(self, pending_id):
        with self._connect() as c:
            c.row_factory = dict_row
            return c.execute(
                "UPDATE pending_actions SET status = 'resolving' WHERE id = %s AND status = 'pending' RETURNING *",
                (pending_id,)).fetchone()

    def release(self, pending_id, error):
        with self._connect() as c:
            c.execute("UPDATE pending_actions SET status = 'pending', last_error = %s"
                      " WHERE id = %s AND status = 'resolving'", (_clean(error), pending_id))

    def finish(self, pending_id, status, decision, edited, result, note):
        with self._connect() as c:
            c.execute(
                "UPDATE pending_actions SET status = %s, decision = %s, edited = %s, result = %s, note = %s,"
                " last_error = '', decided_at = now() WHERE id = %s",
                (status, decision, Jsonb(_clean(edited)) if edited is not None else None,
                 _clean(result), _clean(note), pending_id))
