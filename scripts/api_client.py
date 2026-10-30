import argparse
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentlab import config

BASE, H = "http://127.0.0.1:8010", {"X-API-Key": config.ORCH_API_KEY}


def show(o):
    print(json.dumps(o, indent=2, ensure_ascii=True, default=str))


def call(method, path, **kw):
    r = requests.request(method, BASE + path, headers=H, timeout=30, **kw)
    if r.status_code >= 400:
        sys.exit(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
s = sub.add_parser("submit")
s.add_argument("prompt")
s.add_argument("--wait", action="store_true")
for name in ("status", "run"):
    sub.add_parser(name).add_argument("id")
sub.add_parser("pending")
d = sub.add_parser("decide")
d.add_argument("id", type=int)
d.add_argument("decision", choices=["approve", "edit", "reject"])
d.add_argument("--content")
d.add_argument("--path")
d.add_argument("--overwrite", action="store_true")
d.add_argument("--note", default="")
a = ap.parse_args()

if a.cmd == "submit":
    tid = call("POST", "/tasks", json={"prompt": a.prompt})["task_id"]
    print("task_id:", tid)
    if a.wait:
        t0, last = time.time(), ""
        while time.time() - t0 < 600:
            st = call("GET", f"/tasks/{tid}")
            if st["state"] != last:
                print(f"[{time.time() - t0:4.0f}s] {st['state']}")
                last = st["state"]
            if st["state"] in ("SUCCESS", "FAILURE"):
                show({"result": st["result"], "error": st["error"]})
                break
            time.sleep(2)
elif a.cmd == "status":
    show(call("GET", f"/tasks/{a.id}"))
elif a.cmd == "run":
    show(call("GET", f"/runs/{a.id}"))
elif a.cmd == "pending":
    show(call("GET", "/pending"))
else:
    edits = {k: v for k, v in (("content", a.content), ("path", a.path)) if v is not None}
    if a.overwrite:
        edits["overwrite"] = True
    show(call("POST", f"/pending/{a.id}/decide",
              json={"decision": a.decision, "edits": edits or None, "note": a.note}))
