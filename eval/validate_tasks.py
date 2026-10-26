import json, sys
from collections import Counter
from pathlib import Path

path = Path(__file__).resolve().parent / "tasks.jsonl"
tasks, errors = [], []
for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
        continue
    try:
        t = json.loads(line)
    except json.JSONDecodeError as e:
        errors.append(f"line {i}: invalid JSON ({e})"); continue
    for k in ("id", "split", "category", "prompt", "expect"):
        if k not in t:
            errors.append(f"line {i}: missing '{k}'")
    if t.get("split") not in ("dev", "test"):
        errors.append(f"line {i}: split must be dev|test")
    if t.get("expect", {}).get("outcome") not in ("answer", "refuse", "escalate"):
        errors.append(f"line {i}: bad outcome")
    tasks.append(t)

ids = Counter(t.get("id") for t in tasks)
errors += [f"duplicate id {k}" for k, v in ids.items() if v > 1]
print(f"{len(tasks)} tasks | by category: {dict(Counter(t.get('category') for t in tasks))}")
print(f"by split: {dict(Counter(t.get('split') for t in tasks))}")
if len(tasks) < 30:
    print(f"NOTE: {len(tasks)} < 30 target, keep writing tasks")
for e in errors:
    print("ERROR:", e)
sys.exit(1 if errors else 0)