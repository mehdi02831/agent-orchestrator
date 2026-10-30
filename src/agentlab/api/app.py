import secrets
from contextlib import asynccontextmanager
from typing import Any, Literal

from celery.result import AsyncResult
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

from .. import config
from ..agents.hitl_run import resolve
from ..hitl import HitlError
from ..jobs import celery_app as jobs
from ..memory.pending import PostgresPendingStore
from ..memory.store import PostgresRunStore
from ..tools.base import ToolError


@asynccontextmanager
async def lifespan(_app):
    try:
        PostgresPendingStore.from_config().init_schema()
    except Exception as e:
        print("WARNING: could not initialise the database schema:", type(e).__name__)
    yield


app = FastAPI(title="agentlab", lifespan=lifespan)


def get_run_store():
    return PostgresRunStore.from_config()


def get_pending_store():
    return PostgresPendingStore.from_config()


def require_key(x_api_key: str | None = Header(default=None)):
    key = config.ORCH_API_KEY
    if not key or not x_api_key or not secrets.compare_digest(x_api_key.encode(), key.encode()):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class EchoIn(BaseModel):
    text: str = Field(min_length=1, max_length=500)


class TaskIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)

    @field_validator("prompt")
    @classmethod
    def not_blank(cls, v):
        if not v.strip():
            raise ValueError("prompt must not be blank")
        return v


class DecideIn(BaseModel):
    decision: Literal["approve", "edit", "reject"]
    edits: dict[str, Any] | None = None
    note: str = Field(default="", max_length=500)


def _pending_view(row):
    keys = ("id", "run_id", "status", "prompt", "reason", "reason_type", "proposed", "last_error", "created_at")
    return {k: row.get(k) for k in keys}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/echo", dependencies=[Depends(require_key)])
def echo(body: EchoIn):
    return {"task_id": jobs.echo.delay(body.text).id}


@app.get("/echo/{task_id}", dependencies=[Depends(require_key)])
def echo_status(task_id: str):
    res = AsyncResult(task_id, app=jobs.app)
    return {"task_id": task_id, "state": res.state, "result": res.result if res.successful() else None}


@app.post("/tasks", dependencies=[Depends(require_key)])
def submit_task(body: TaskIn):
    return {"task_id": jobs.run_task.delay(body.prompt).id}


@app.get("/tasks/{task_id}", dependencies=[Depends(require_key)])
def task_status(task_id: str):
    res = AsyncResult(task_id, app=jobs.app)
    out = {"task_id": task_id, "state": res.state, "result": None, "error": None, "run": None}
    if res.state == "SUCCESS" and isinstance(res.result, dict):
        out["result"] = res.result
        if res.result.get("run_id"):
            out["run"] = get_run_store().get_run(res.result["run_id"])
    elif res.state == "FAILURE":
        out["error"] = str(res.result)[:200]
    return out


@app.get("/runs/{run_id}", dependencies=[Depends(require_key)])
def get_run(run_id: int):
    run = get_run_store().get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="no such run")
    return run


@app.get("/pending", dependencies=[Depends(require_key)])
def list_pending():
    return {"pending": [_pending_view(r) for r in get_pending_store().list_open()]}


@app.post("/pending/{pending_id}/decide", dependencies=[Depends(require_key)])
def decide_pending(pending_id: int, body: DecideIn):
    try:
        status, msg = resolve(get_run_store(), get_pending_store(), pending_id, body.decision,
                              edits=body.edits, note=body.note)
    except HitlError as e:
        raise HTTPException(status_code=409 if "resolved" in str(e) else 400, detail=str(e))
    except ToolError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"id": pending_id, "status": status, "message": msg}
