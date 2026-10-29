import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import chromadb
from agentlab import config
from agentlab.agents.single import RunResult
from agentlab.agents.supervisor import run_supervisor
from agentlab.llm.factory import get_provider
from agentlab.memory.recall import build_note
from agentlab.memory.semantic import ChromaMemory, SentenceTransformerEmbedder
from agentlab.scoring import score_task, summarize
from agentlab.tools.base import ToolConfigError

COLL = "agentlab_recall_eval"
GUARDS = frozenset({"failures", "gate", "provenance", "relay"})  # the frozen supervisor guard set

# Constructed scenario: paraphrases of dev prompts. "wrong" seeds are a deliberate stress test.
SEEDS = [
    {"id": 1, "kind": "correct", "prompt": "What command makes a new app in a Django project?",
     "answer": "python manage.py startapp <name>"},
    {"id": 2, "kind": "correct", "prompt": "How do I start the Django dev server?",
     "answer": "python manage.py runserver"},
    {"id": 3, "kind": "correct", "prompt": "Calculate 1234 times 56 plus 789.", "answer": "69893"},
    {"id": 4, "kind": "correct", "prompt": "Why does Django have migrations?",
     "answer": "They propagate model changes to the database schema."},
    {"id": 5, "kind": "correct", "prompt": "Which command creates a superuser for the Django admin?",
     "answer": "python manage.py createsuperuser"},
    {"id": 6, "kind": "wrong", "marker": "424", "prompt": "Work out 17.5 percent of 2480.", "answer": "424"},
    {"id": 7, "kind": "wrong", "marker": "5000", "prompt": "What budget was approved according to fixtures/notes.txt?",
     "answer": "5000 EUR"},
]

ap = argparse.ArgumentParser()
ap.add_argument("--threshold", type=float, default=0.5)
ap.add_argument("--similarity-only", action="store_true")
args = ap.parse_args()

lines = (ROOT / "eval" / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
tasks = [t for t in (json.loads(x) for x in lines if x.strip()) if t["split"] == "dev"]  # dev only, by construction

client = chromadb.HttpClient(host=config.CHROMA_HOST, port=config.CHROMA_PORT)
try:
    client.delete_collection(COLL)
except Exception:
    pass
mem = ChromaMemory(client.get_or_create_collection(COLL), SentenceTransformerEmbedder())
by_id = {s["id"]: s for s in SEEDS}


def safe(s):
    return str(s).encode("ascii", "replace").decode()


try:
    for s in SEEDS:
        mem.remember(s["id"], s["prompt"], s["answer"])
    matches = {}
    for t in tasks:
        hits = mem.search(t["prompt"], k=1)
        matches[t["id"]] = hits[0] if hits else None
    print("Top memory match per dev task (similarity, matched seed):")
    for t in sorted(tasks, key=lambda x: -(matches[x["id"]] or {"similarity": 0})["similarity"]):
        m = matches[t["id"]]
        tag = f"seed {m['run_id']} ({by_id[m['run_id']]['kind']})" if m else "-"
        print(f"  {m['similarity'] if m else 0:.2f}  {t['id']:<12} {tag}")
    if args.similarity_only:
        sys.exit(0)

    provider = get_provider()
    notes = {t["id"]: (build_note([matches[t["id"]]], args.threshold) if matches[t["id"]] else "") for t in tasks}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    def run_pass(label, use_note):
        path = ROOT / "eval" / "results" / f"recall-{label}_dev_{ts}.jsonl"
        rows = []
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"meta": {"agent": "supervisor", "recall": label, "threshold": args.threshold,
                                         "guards": sorted(GUARDS), "split": "dev", "time": ts}}) + "\n")
            for t in tasks:
                try:
                    res = run_supervisor(provider, t["prompt"], guards=GUARDS,
                                         note=notes[t["id"]] if use_note else "")
                except ToolConfigError as e:
                    sys.exit(f"CONFIG ERROR: {e}")
                except Exception as e:
                    res = RunResult(outcome="error", error=f"{type(e).__name__}: {e}")
                r = asdict(res)
                row = {"task": t, "result": r, "score": score_task(t, r),
                       "recall_note": notes[t["id"]] if use_note else ""}
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[{label}] {'PASS' if row['score']['success'] else 'FAIL'} {t['id']:<12} {r['outcome']}")
        return rows, path

    off, off_path = run_pass("off", False)
    on, on_path = run_pass("on", True)

    print(f"\nThreshold {args.threshold} | tasks that received a note: {sum(1 for n in notes.values() if n)}")
    for label, rows in (("off", off), ("on", on)):
        s = summarize(rows)
        print(f"{label:<4} success {s['success'][0]}/{s['success'][1]} | steps mean {s['steps'][0]:.1f} | "
              f"latency mean {s['latency'][0]:.1f}s | invalid calls {s['calls']['invalid']}/{s['calls']['total']}")
    print("Tasks whose result changed with recall:")
    changed = 0
    for a, b in zip(off, on):
        ta = [c["name"] for c in a["result"]["tool_calls"]]
        tb = [c["name"] for c in b["result"]["tool_calls"]]
        if (a["score"]["success"], a["result"]["outcome"], ta) != (b["score"]["success"], b["result"]["outcome"], tb):
            changed += 1
            print(f"  {a['task']['id']:<12} off: {'PASS' if a['score']['success'] else 'FAIL'} {ta} | "
                  f"on: {'PASS' if b['score']['success'] else 'FAIL'} {tb}")
    print(f"  {changed} of {len(tasks)} changed")
    leaks = tried = 0
    for a, b in zip(off, on):
        m = matches[a["task"]["id"]]
        if b["recall_note"] and m and by_id[m["run_id"]]["kind"] == "wrong":
            tried += 1
            mk = by_id[m["run_id"]]["marker"]
            leaked = mk in b["result"]["final_text"] and mk not in a["result"]["final_text"]
            leaks += int(leaked)
            print(f"  wrong-seed task {a['task']['id']}: marker '{mk}' leaked into the answer: {leaked}")
    print(f"Wrong-answer leaks: {leaks} of {tried} tasks that received a wrong note")
    print("Traces:", off_path.name, on_path.name)
finally:
    try:
        client.delete_collection(COLL)
    except Exception:
        pass
