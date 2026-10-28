import argparse
import json
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "eval" / "results"


def safe(s):
    return str(s).encode("ascii", "replace").decode()


ap = argparse.ArgumentParser()
ap.add_argument("ids", nargs="+")
ap.add_argument("--file")
args = ap.parse_args()

path = Path(args.file) if args.file else max(RES.glob("single_dev_*.jsonl"), key=lambda p: p.stat().st_mtime)
print("file:", path.name)
for line in path.read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    if "task" not in row or row["task"]["id"] not in args.ids:
        continue
    t, r, s = row["task"], row["result"], row["score"]
    print("=" * 70)
    print(t["id"], "| expected:", t["expect"]["outcome"], "| success:", s["success"], "| outcome:", r["outcome"])
    print("PROMPT:", safe(t["prompt"]))
    for c in r["tool_calls"]:
        print(f"  CALL {c['name']} {safe(json.dumps(c['arguments']))} -> [{c['kind']}] {safe(c['content'])}")
    print("FINAL:", safe(r["final_text"] or r["escalation_reason"] or r["error"]))
