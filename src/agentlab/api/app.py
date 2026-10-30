import secrets

from celery.result import AsyncResult
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .. import config
from ..jobs import celery_app as jobs

app = FastAPI(title="agentlab")


def require_key(x_api_key: str | None = Header(default=None)):
    key = config.ORCH_API_KEY
    if not key or not x_api_key or not secrets.compare_digest(x_api_key.encode(), key.encode()):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class EchoIn(BaseModel):
    text: str = Field(min_length=1, max_length=500)


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
