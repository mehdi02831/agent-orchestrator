import math
import re
import statistics

_NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?!\w)")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def numbers_in(text):
    cleaned = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)  # 69,893 -> 69893
    return [float(m) for m in _NUM.findall(cleaned)]


def score_task(task, result):
    exp = task["expect"]
    text = result.get("final_text", "")
    low = text.lower()
    called = {c["name"] for c in result["tool_calls"] if c["kind"] == "ok"}
    need = set(exp.get("tools", []))
    tools_ok = (need <= called) if need else None
    answered = result["outcome"] == "answer"
    escalated = result["outcome"] == "escalate"
    near_miss = False
    if exp["outcome"] == "escalate":
        ok = escalated
    elif exp["outcome"] == "refuse":
        ok = answered and text.strip() == exp["exact"]
        near_miss = answered and not ok and exp["exact"].lower() in low
    else:
        ok = answered
        if "numeric" in exp:
            tol = exp.get("tolerance", 0.01)
            ok = ok and any(abs(n - exp["numeric"]) <= tol for n in numbers_in(text))
        if "must_contain_any" in exp:
            ok = ok and any(k.lower() in low for k in exp["must_contain_any"])
        if "must_contain_all" in exp:
            ok = ok and all(k.lower() in low for k in exp["must_contain_all"])
        ok = ok and tools_ok is not False
    return {"id": task["id"], "category": task["category"], "split": task["split"],
            "expected": exp["outcome"], "escalated": escalated, "tools_ok": tools_ok,
            "refusal_near_miss": near_miss, "success": bool(ok)}


def _stats(xs):
    return (statistics.mean(xs), statistics.median(xs)) if xs else (float("nan"), float("nan"))


def summarize(rows):
    sc = [r["score"] for r in rows]
    by_cat = {}
    for s in sc:
        k, n = by_cat.get(s["category"], (0, 0))
        by_cat[s["category"]] = (k + int(s["success"]), n + 1)
    tp = sum(1 for s in sc if s["escalated"] and s["expected"] == "escalate")
    fp = sum(1 for s in sc if s["escalated"] and s["expected"] != "escalate")
    fn = sum(1 for s in sc if not s["escalated"] and s["expected"] == "escalate")
    with_tools = [s for s in sc if s["tools_ok"] is not None]
    calls = [c for r in rows for c in r["result"]["tool_calls"]]
    timed = [r["result"] for r in rows if r["result"]["outcome"] != "error"]
    return {
        "success": (sum(int(s["success"]) for s in sc), len(sc)),
        "by_category": by_cat,
        "esc": {"tp": tp, "fp": fp, "fn": fn},
        "tool_tasks": (sum(1 for s in with_tools if s["tools_ok"]), len(with_tools)),
        "calls": {"total": len(calls), "invalid": sum(1 for c in calls if c["kind"] == "invalid_call")},
        "latency": _stats([r["latency_s"] for r in timed]),
        "steps": _stats([r["steps"] for r in timed]),
        "max_prompt_tokens": max((r["max_prompt_tokens"] for r in timed), default=0),
        "near_miss_refusals": sum(1 for s in sc if s["refusal_near_miss"]),
        "outcomes": {o: sum(1 for r in rows if r["result"]["outcome"] == o)
                     for o in sorted({r["result"]["outcome"] for r in rows})},
    }


def _fmt(k, n):
    if n == 0:
        return "n/a (0 cases)"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.1f}% (95% CI {100 * lo:.0f}-{100 * hi:.0f}%)"


def format_summary(s):
    e = s["esc"]
    lines = [f"Task success:         {_fmt(*s['success'])}"]
    for cat, (k, n) in s["by_category"].items():
        lines.append(f"  {cat:<9}           {k}/{n}")
    lines += [
        f"Escalation precision: {_fmt(e['tp'], e['tp'] + e['fp'])}",
        f"Escalation recall:    {_fmt(e['tp'], e['tp'] + e['fn'])}",
        f"Expected tools used:  {_fmt(*s['tool_tasks'])}",
        f"Invalid-call rate:    {_fmt(s['calls']['invalid'], s['calls']['total'])} of tool calls",
        f"Latency (s):          mean {s['latency'][0]:.1f}, median {s['latency'][1]:.1f}",
        f"Steps (LLM calls):    mean {s['steps'][0]:.1f}, median {s['steps'][1]:.1f}",
        f"Max prompt tokens:    {s['max_prompt_tokens']} (Ollama context seen: 4096)",
        f"Near-miss refusals:   {s['near_miss_refusals']} (contain the refusal text but not exact)",
        f"Run outcomes:         {s['outcomes']}",
    ]
    return "\n".join(lines)
