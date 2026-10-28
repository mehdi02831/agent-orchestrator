import json
import sys
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "eval" / "results"


def load(p):
    rows = {}
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "task" in row:
            rows[row["task"]["id"]] = row
    return rows


if len(sys.argv) == 3:
    pa, pb = Path(sys.argv[1]), Path(sys.argv[2])
else:
    files = sorted(RES.glob("*_dev_*.jsonl"), key=lambda p: p.stat().st_mtime)
    if len(files) < 2:
        sys.exit("Need at least two runs in eval/results")
    pa, pb = files[-2], files[-1]

a, b = load(pa), load(pb)
common = [i for i in a if i in b]
print("A:", pa.name, "| B:", pb.name, "|", len(common), "common tasks")
diffs = 0
for i in common:
    ra, rb = a[i]["result"], b[i]["result"]
    sa, sb = a[i]["score"]["success"], b[i]["score"]["success"]
    ta, tb = [c["name"] for c in ra["tool_calls"]], [c["name"] for c in rb["tool_calls"]]
    if (sa, ra["outcome"], ta) != (sb, rb["outcome"], tb):
        diffs += 1
        print(f"{i:<12} A: {'PASS' if sa else 'FAIL'} {ra['outcome']} {ta}")
        print(f"{'':<12} B: {'PASS' if sb else 'FAIL'} {rb['outcome']} {tb}")
print(f"passed: A {sum(a[i]['score']['success'] for i in common)} | B {sum(b[i]['score']['success'] for i in common)}")
print(f"{diffs} of {len(common)} tasks differ in success, outcome or tool sequence")
