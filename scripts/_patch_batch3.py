import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EDITS = {
    "src/agentlab/scoring.py": [
        ("import statistics\n",
         "import statistics\n\nfrom . import config\n"),
        ('    return {"id": task["id"], "category": task["category"], "split": task["split"],\n',
         """    refuse_failure = None
    if exp["outcome"] == "refuse" and not ok:
        rag = [c for c in result["tool_calls"] if c["name"] == "rag_ask" and c["kind"] == "ok"]
        if not rag:
            refuse_failure = "skipped_rag"
        elif any(config.REFUSAL_TEXT in c.get("content", "") for c in rag):
            refuse_failure = "not_relayed"
        else:
            refuse_failure = "rag_answered"
    return {"id": task["id"], "category": task["category"], "split": task["split"],
"""),
        ('            "refusal_near_miss": near_miss, "success": bool(ok)}',
         '            "refusal_near_miss": near_miss, "refuse_failure": refuse_failure, "success": bool(ok)}'),
        ('    with_tools = [s for s in sc if s["tools_ok"] is not None]\n',
         """    with_tools = [s for s in sc if s["tools_ok"] is not None]
    rf = {}
    for s in sc:
        if s.get("refuse_failure"):
            rf[s["refuse_failure"]] = rf.get(s["refuse_failure"], 0) + 1
"""),
        ('        "near_miss_refusals": sum(1 for s in sc if s["refusal_near_miss"]),\n',
         '        "near_miss_refusals": sum(1 for s in sc if s["refusal_near_miss"]),\n        "refuse_failures": rf,\n'),
        ("        f\"Run outcomes:         {s['outcomes']}\",\n",
         "        f\"Refuse failures:      {s['refuse_failures'] or 'none'}\",\n        f\"Run outcomes:         {s['outcomes']}\",\n"),
    ],
    "eval/run_eval.py": [
        ("from agentlab.agents.single import RunResult, run_single",
         "from agentlab.agents.single import PROMPTS, RunResult, run_single"),
        ('ap.add_argument("--ids", nargs="+")\n',
         'ap.add_argument("--ids", nargs="+")\nap.add_argument("--prompt", choices=sorted(PROMPTS), default="v1")\n'),
        ('res = run_single(provider, t["prompt"])',
         'res = run_single(provider, t["prompt"], prompt_version=args.prompt)'),
        ('"split": args.split, "time": ts, "n_tasks": len(tasks)}}',
         '"split": args.split, "prompt": args.prompt, "time": ts, "n_tasks": len(tasks)}}'),
        ('print("\\nTraces:", out_path)',
         'print("\\nPrompt:", args.prompt, "| Traces:", out_path)'),
    ],
    "tests/test_tools.py": [
        ("class _Page:\n    def __init__(self, code, text):\n        self.status_code, self.text = code, text\n",
         "class _Page:\n    def __init__(self, code, text, headers=None):\n        self.status_code, self.text, self.headers = code, text, headers or {}\n"),
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
