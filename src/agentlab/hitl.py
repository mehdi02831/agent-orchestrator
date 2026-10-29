from .tools.base import ToolError
from .tools.write import validate_write, write_file

DECISIONS = ("approve", "edit", "reject")
EDIT_KEYS = {"path", "content", "overwrite"}


class HitlError(Exception):
    pass


def propose(store, path, content, prompt="", reason="", reason_type="risky_action",
            run_id=None, state=None, overwrite=False) -> int:
    """Queue a write for human review. Invalid paths and content are refused up front (ToolError)."""
    validate_write(path, content)
    return store.create({
        "run_id": run_id, "prompt": prompt, "reason": reason, "reason_type": reason_type,
        "proposed": {"action": "write_file", "path": path, "content": content, "overwrite": bool(overwrite)},
        "state": state or {}})


def execute_action(proposed: dict) -> str:
    if proposed.get("action") != "write_file":
        raise ToolError(f"unknown action: {proposed.get('action')}")
    return write_file(proposed.get("path", ""), proposed.get("content", ""), bool(proposed.get("overwrite", False)))


def decide(store, pending_id, decision, edits=None, note=""):
    """Returns (status, message). status is approved, rejected, or pending (execution failed, retry allowed)."""
    if decision not in DECISIONS:
        raise HitlError(f"decision must be one of {list(DECISIONS)}")
    if decision == "edit":
        if not edits:
            raise HitlError("edit needs at least one change")
        if set(edits) - EDIT_KEYS:
            raise HitlError(f"unknown edit keys: {sorted(set(edits) - EDIT_KEYS)}")
    elif edits:
        raise HitlError("changes are only allowed with decision 'edit'")
    row = store.claim(pending_id)
    if row is None:
        raise HitlError("not found or already resolved")
    if decision == "reject":
        store.finish(pending_id, "rejected", decision, None, "", note)
        return "rejected", "nothing was executed"
    final = dict(row["proposed"])
    if decision == "edit":
        final.update(edits)
    try:
        result = execute_action(final)
    except ToolError as e:
        store.release(pending_id, str(e))
        return "pending", str(e)
    except Exception:
        store.release(pending_id, "unexpected error")
        raise
    store.finish(pending_id, "approved", decision, final if decision == "edit" else None, result, note)
    return "approved", result
