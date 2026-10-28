import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EDITS = {
    "src/agentlab/agents/single.py": [
        ("from ..tools.registry import execute, tool_schemas\n",
         "from ..tools.registry import execute, tool_schemas\nfrom .guardrails import is_risky, refusal_relay, unrecovered_failure\n"),
        ('    error: str = ""\n',
         '    error: str = ""\n    guard_fired: str = ""\n'),
        ('prompt_version: str = "v1") -> RunResult:',
         'prompt_version: str = "v1", guards: frozenset = frozenset()) -> RunResult:'),
        ('    res = RunResult(outcome="max_steps")\n    t0 = time.perf_counter()\n',
         """    res = RunResult(outcome="max_steps")
    t0 = time.perf_counter()
    if "gate" in guards and is_risky(prompt):
        reason = "guard: request to delete, overwrite or write files"
        res.outcome, res.escalation_reason, res.guard_fired = "escalate", reason, "gate"
        res.tool_calls.append({"name": "ask_human", "arguments": {"reason": reason}, "kind": "ok", "content": ""})
        res.latency_s = time.perf_counter() - t0
        return res
"""),
        ("    res.latency_s = time.perf_counter() - t0\n    return res\n",
         """    if res.outcome == "answer":
        if "relay" in guards and refusal_relay(res.tool_calls) and res.final_text.strip() != config.REFUSAL_TEXT:
            res.final_text, res.guard_fired = config.REFUSAL_TEXT, "relay"
        elif "failures" in guards and unrecovered_failure(res.tool_calls):
            res.outcome, res.escalation_reason, res.guard_fired = "escalate", "guard: tool failure not recovered", "failures"
    res.latency_s = time.perf_counter() - t0
    return res
"""),
    ],
    "eval/run_eval.py": [
        ('ap.add_argument("--prompt", choices=sorted(PROMPTS), default="v1")\n',
         'ap.add_argument("--prompt", choices=sorted(PROMPTS), default="v1")\n'
         'ap.add_argument("--guards", nargs="*", choices=["gate", "relay", "failures"], default=[])\n'),
        ('res = run_single(provider, t["prompt"], prompt_version=args.prompt)',
         'res = run_single(provider, t["prompt"], prompt_version=args.prompt, guards=frozenset(args.guards))'),
        ('"split": args.split, "prompt": args.prompt, "time": ts, "n_tasks": len(tasks)}}',
         '"split": args.split, "prompt": args.prompt, "guards": sorted(args.guards), "time": ts, "n_tasks": len(tasks)}}'),
        ('print("\\n" + format_summary(summarize(rows)))',
         """print("\\n" + format_summary(summarize(rows)))
fired = {}
for row_ in rows:
    g_ = row_["result"].get("guard_fired")
    if g_:
        fired[g_] = fired.get(g_, 0) + 1
print("Guards enabled:      ", sorted(args.guards) or "none", "| fired:", fired or "none")"""),
    ],
}

texts = {}
for rel, pairs in EDITS.items():
    t = (ROOT / rel).read_text(encoding="utf-8")
    for old, new in pairs:
        n = t.count(old)
        if n != 1:
            sys.exit(f"ABORT, nothing written: {n} matches in {rel} for {old[:70]!r}")
        t = t.replace(old, new)
    texts[rel] = t
for rel, t in texts.items():
    (ROOT / rel).write_text(t, encoding="utf-8")
print("patched:", ", ".join(texts))
