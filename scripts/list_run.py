import json, sys
from pathlib import Path
RES = Path(__file__).resolve().parents[1] / "eval" / "results"
p = Path(sys.argv[1]) if len(sys.argv) > 1 else max(RES.glob("single_dev_*.jsonl"), key=lambda x: x.stat().st_mtime)
print(p.name)
for line in p.read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    if "meta" in row:
        m = row["meta"]
        print("config: prompt", m.get("prompt", "v1"), "| guards", m.get("guards", []))
    if "task" in row:
        r, s = row["result"], row["score"]
        g = r.get("guard_fired") or "-"
        print(f"{'PASS' if s['success'] else 'FAIL'} {row['task']['id']:<12} {r['outcome']:<9} guard={g:<9} {[c['name'] for c in r['tool_calls']]}")
