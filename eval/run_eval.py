import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentlab import config
from agentlab.agents.single import PROMPTS, RunResult, run_single
from agentlab.llm.factory import get_provider
from agentlab.scoring import format_summary, score_task, summarize
from agentlab.tools.base import ToolConfigError


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def ascii_safe(s):
    return s.encode("ascii", "replace").decode()


ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
ap.add_argument("--ids", nargs="+")
ap.add_argument("--prompt", choices=sorted(PROMPTS), default="v1")
ap.add_argument("--guards", nargs="*", choices=["gate", "relay", "failures"], default=[])
ap.add_argument("--final", action="store_true", help="required to touch the frozen test split")
args = ap.parse_args()

if args.split in ("test", "all") and not args.final:
    sys.exit("The test split is frozen. Re-run with --final only for the single final evaluation.")

lines = (ROOT / "eval" / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
tasks = [json.loads(x) for x in lines if x.strip()]
if args.split != "all":
    tasks = [t for t in tasks if t["split"] == args.split]
if args.ids:
    tasks = [t for t in tasks if t["id"] in args.ids]
if not tasks:
    sys.exit("No tasks selected.")

provider = get_provider()
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = ROOT / "eval" / "results" / f"single_{args.split}_{ts}.jsonl"
meta = {"meta": {"agent": "single", "model": config.OLLAMA_MODEL, "git": git_rev(),
                 "split": args.split, "prompt": args.prompt, "guards": sorted(args.guards), "time": ts, "n_tasks": len(tasks)}}
rows, consecutive_errors = [], 0
with out_path.open("w", encoding="utf-8") as f:
    f.write(json.dumps(meta) + "\n")
    for t in tasks:
        try:
            res = run_single(provider, t["prompt"], prompt_version=args.prompt, guards=frozenset(args.guards))
            consecutive_errors = 0
        except ToolConfigError as e:
            sys.exit(f"CONFIG ERROR, aborting the run: {e}")
        except Exception as e:  # LLM or network failure
            res = RunResult(outcome="error", error=f"{type(e).__name__}: {e}")
            consecutive_errors += 1
        r = asdict(res)
        sc = score_task(t, r)
        rows.append({"task": t, "result": r, "score": sc})
        f.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
        f.flush()
        tools = ",".join(c["name"] for c in r["tool_calls"]) or "-"
        print(f"[{'PASS' if sc['success'] else 'FAIL'}] {t['id']:<12} {r['outcome']:<9} "
              f"steps={r['steps']} {r['latency_s']:.1f}s tools={tools}")
        if not sc["success"]:
            print("        ->", ascii_safe((r["final_text"] or r["escalation_reason"] or r["error"])[:140]))
        if consecutive_errors >= 3:
            sys.exit("3 consecutive errors (is Ollama running?). Partial results saved: " + str(out_path))

print("\n" + format_summary(summarize(rows)))
fired = {}
for row_ in rows:
    g_ = row_["result"].get("guard_fired")
    if g_:
        fired[g_] = fired.get(g_, 0) + 1
print("Guards enabled:      ", sorted(args.guards) or "none", "| fired:", fired or "none")
print("\nPrompt:", args.prompt, "| Traces:", out_path)
