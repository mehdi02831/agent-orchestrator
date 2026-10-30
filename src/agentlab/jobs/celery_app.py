import socket

from celery import Celery

from .. import config

app = Celery("agentlab", broker=config.REDIS_URL, backend=config.REDIS_URL)
app.conf.update(result_expires=3600, task_track_started=True, worker_prefetch_multiplier=1,
                task_acks_late=True, task_reject_on_worker_lost=True)


@app.task(name="agentlab.echo")
def echo(text: str) -> dict:
    """Infrastructure check: proves the worker runs in Docker and can be reached through Redis."""
    return {"echo": text, "worker_host": socket.gethostname()}


@app.task(name="agentlab.run_task", soft_time_limit=900, time_limit=960)
def run_task(prompt: str) -> dict:
    from .runner import run_and_record
    return run_and_record(prompt)
