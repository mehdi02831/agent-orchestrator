from dataclasses import asdict, is_dataclass


def _clean(x):
    """Postgres text and jsonb reject NUL characters; strip them recursively."""
    if isinstance(x, str):
        return x.replace("\x00", "")
    if isinstance(x, dict):
        return {_clean(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def run_record(agent, prompt, result, config_meta=None) -> dict:
    """Flatten a RunResult / SupervisorResult (or a plain dict) into one storable record."""
    r = asdict(result) if is_dataclass(result) else dict(result)
    rec = {
        "agent": agent, "prompt": prompt, "config": config_meta or {},
        "status": r.get("outcome", ""), "final_text": r.get("final_text", ""),
        "escalation_reason": r.get("escalation_reason", ""), "reason_type": r.get("reason_type", ""),
        "guard_fired": r.get("guard_fired", ""), "error": r.get("error", ""),
        "steps": int(r.get("steps", 0)), "invalid_plans": int(r.get("invalid_plans", 0)),
        "latency_s": float(r.get("latency_s", 0.0)), "max_prompt_tokens": int(r.get("max_prompt_tokens", 0)),
        "tool_calls": list(r.get("tool_calls", [])), "delegations": list(r.get("delegations", [])),
    }
    return _clean(rec)
