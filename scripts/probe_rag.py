import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import requests
from agentlab import config

QUESTIONS = [
    "How do I create a new Django app inside a project?",
    "What does the Django ORM method select_related do?",
    "What is the capital of Australia?",
]
for q in QUESTIONS:
    r = requests.post(f"{config.RAG_API_URL}/ask", json={"question": q},
    headers={"X-API-Key": config.RAG_API_KEY}, timeout=120)
    print("=" * 60)
    print("Q:", q, "|", r.status_code, f"{r.elapsed.total_seconds():.2f}s")
    print(json.dumps(r.json(), indent=2, ensure_ascii=False))
    if isinstance(r.json(), dict):
        print("exact refusal match:", r.json().get("answer") == config.REFUSAL_TEXT)
