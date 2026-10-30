from ..agents.hitl_run import FROZEN_GUARDS, run_with_review
from ..llm.factory import get_provider
from ..memory.pending import PostgresPendingStore
from ..memory.records import run_record
from ..memory.store import PostgresRunStore


def execute_task(prompt, provider, run_store, pending_store) -> dict:
    """Run one task with the frozen supervisor guards. On any error, save an 'error' run and re-raise."""
    try:
        out = run_with_review(provider, prompt, run_store, pending_store)
    except Exception as e:
        try:
            run_store.save_run(run_record("supervisor", prompt, {"outcome": "error", "error": type(e).__name__},
                                          {"guards": sorted(FROZEN_GUARDS), "review": True}))
        except Exception:
            pass  # never mask the original error
        raise
    res = out.result
    return {"run_id": out.run_id, "outcome": res.outcome, "proposal": out.proposal,
            "pending_id": out.pending_id, "final": (res.final_text or res.escalation_reason)[:500]}


def run_and_record(prompt) -> dict:
    pending = PostgresPendingStore.from_config()
    pending.init_schema()  # also creates the run tables
    return execute_task(prompt, get_provider(), PostgresRunStore.from_config(), pending)
