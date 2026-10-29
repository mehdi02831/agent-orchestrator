import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentlab.agents.hitl_run import run_with_review
from agentlab.memory.pending import PostgresPendingStore
from agentlab.llm.factory import get_provider
from agentlab.memory.records import run_record
from agentlab.memory.semantic import ChromaMemory, should_remember
from agentlab.memory.store import PostgresRunStore

GUARDS = ["failures", "gate", "provenance", "relay"]  # the frozen supervisor guard set


def safe(s):
    return str(s).encode("ascii", "replace").decode()


ap = argparse.ArgumentParser()
ap.add_argument("prompt", nargs="?")
ap.add_argument("--show", type=int, help="print a stored run")
ap.add_argument("--list", action="store_true", help="list recent runs")
ap.add_argument("--search", help="semantic search over remembered answers")
ap.add_argument("--forget", type=int, help="delete a run from Postgres and from semantic memory")
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
elif args.search:
    memory = ChromaMemory.from_config()
    hits = memory.search(args.search, k=3)
    print(f"{memory.count()} remembered runs")
    for h in hits:
        print(f"  run {h['run_id']} sim={h['similarity']:.2f} | {safe(h['prompt'][:50])} -> {safe(h['answer'][:60])}")
    if not hits:
        print("  no matches")
elif args.forget:
    deleted = store.delete_run(args.forget)
    print("postgres:", "deleted" if deleted else "no such run")
    try:
        ChromaMemory.from_config().forget(args.forget)
        print("chroma: forgotten")
    except Exception as e:
        print("chroma: WARNING", type(e).__name__, "(entry may remain)")
elif args.prompt:
    pending = PostgresPendingStore.from_config()
    pending.init_schema()
    out = run_with_review(get_provider(), args.prompt, store, pending, guards=frozenset(GUARDS))
    res, rid = out.result, out.run_id
    print("saved as run", rid, "|", res.outcome, "|", f"{res.latency_s:.1f}s")
    print("final:", safe(res.final_text or res.escalation_reason))
    print("review:", out.proposal, "| pending action", out.pending_id or "-")
    if should_remember(res.outcome, res.final_text):
        try:
            ChromaMemory.from_config().remember(rid, args.prompt, res.final_text)
            print("memory: stored")
        except Exception as e:
            print("memory: WARNING not stored,", type(e).__name__)
    else:
        print("memory: skipped (not a plain answer)")
else:
    ap.print_help()
