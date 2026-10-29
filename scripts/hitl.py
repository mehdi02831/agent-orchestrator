import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentlab.hitl import HitlError, decide, propose
from agentlab.memory.pending import PostgresPendingStore
from agentlab.tools.base import ToolError


def safe(s):
    return str(s).encode("ascii", "replace").decode()


ap = argparse.ArgumentParser(description="Human review of paused write actions")
sub = ap.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("propose", help="queue a write for review (manual stand-in until the agent does it)")
p.add_argument("path")
p.add_argument("content")
sub.add_parser("list")
for name in ("show", "approve", "reject", "edit"):
    s = sub.add_parser(name)
    s.add_argument("id", type=int)
    if name != "show":
        s.add_argument("--note", default="")
    if name == "edit":
        s.add_argument("--content")
        s.add_argument("--path")
        s.add_argument("--overwrite", action="store_true")
args = ap.parse_args()

store = PostgresPendingStore.from_config()
store.init_schema()
try:
    if args.cmd == "propose":
        pid = propose(store, args.path, args.content, prompt="manual", reason="manual proposal")
        print("queued as", pid)
    elif args.cmd == "list":
        rows = store.list_open()
        for r in rows:
            print(r["id"], r["status"], safe(r["proposed"]["path"]), "|", safe(r["reason"]))
        print(len(rows), "pending")
    elif args.cmd == "show":
        r = store.get(args.id)
        if r is None:
            sys.exit("not found")
        print("id", r["id"], "|", r["status"], "|", r["decision"] or "-", "| run", r["run_id"] or "-")
        print("path:", safe(r["proposed"]["path"]), "| overwrite:", r["proposed"].get("overwrite"))
        print("content:", safe(r["proposed"]["content"][:200]))
        if r["last_error"]:
            print("last error:", safe(r["last_error"]))
        if r["result"]:
            print("result:", safe(r["result"]))
    else:
        edits = None
        if args.cmd == "edit":
            edits = {k: v for k, v in (("content", args.content), ("path", args.path)) if v is not None}
            if args.overwrite:
                edits["overwrite"] = True
        status, msg = decide(store, args.id, args.cmd, edits=edits, note=args.note)
        print(status, "|", safe(msg))
except (HitlError, ToolError) as e:
    sys.exit(f"refused: {e}")
