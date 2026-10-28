import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentlab.agents.supervisor import run_supervisor
from agentlab.llm.factory import get_provider
from agentlab.memory.records import run_record
from agentlab.memory.store import PostgresRunStore

GUARDS = ["failures", "gate", "provenance", "relay"]  # the frozen supervisor guard set


def safe(s):
    return str(s).encode("ascii", "replace").decode()


ap = argparse.ArgumentParser()
ap.add_argument("prompt", nargs="?")
ap.add_argument("--show", type=int, help="print a stored run")
ap.add_argument("--list", action="store_true", help="list recent runs")
args = ap.parse_args()

store = PostgresRunStore.from_config()
store.init_schema()
if args.list:
    for r in store.list_runs(10):
        print(r["id"], r["created_at"].strftime("%H:%M:%S"), r["agent"], r["status"], safe(r["prompt"][:60]))
elif args.show:
    r = store.get_run(args.show)
    if r is None:
        sys.exit("no such run")
    print("run", r["id"], "|", r["agent"], "|", r["status"], "| steps", r["steps"], "| guard", r["guard_fired"] or "-")
    print("prompt:", safe(r["prompt"]))
    for d in r["delegations"]:
        print("  delegate", d["agent"], "->", safe(d["result"][:100]))
    for t in r["tool_calls"]:
        print("  tool", t["name"], "[" + t["kind"] + "]")
    print("final:", safe(r["final_text"] or r["escalation_reason"]))
elif args.prompt:
    res = run_supervisor(get_provider(), args.prompt, guards=frozenset(GUARDS))
    rid = store.save_run(run_record("supervisor", args.prompt, res, {"guards": GUARDS}))
    print("saved as run", rid, "|", res.outcome, "|", f"{res.latency_s:.1f}s")
    print("final:", safe(res.final_text or res.escalation_reason))
else:
    ap.print_help()
