import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentlab import config

BASE, H = "http://127.0.0.1:8010", {"X-API-Key": config.ORCH_API_KEY}
print("health:", requests.get(f"{BASE}/health", timeout=5).json())
print("no key ->", requests.post(f"{BASE}/echo", json={"text": "x"}, timeout=5).status_code)
tid = requests.post(f"{BASE}/echo", json={"text": "hello queue"}, headers=H, timeout=5).json()["task_id"]
for _ in range(20):
    st = requests.get(f"{BASE}/echo/{tid}", headers=H, timeout=5).json()
    if st["state"] in ("SUCCESS", "FAILURE"):
        break
    time.sleep(0.5)
print("task:", st)
