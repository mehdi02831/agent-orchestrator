import json
import requests
from .. import config
from .base import ToolError, ToolConfigError


def rag_ask(question: str) -> str:
    if not config.RAG_API_KEY:
        raise ToolConfigError("RAG_API_KEY is not set in .env")
    try:
        r = requests.post(f"{config.RAG_API_URL}/ask", json={"question": question},
                          headers={"X-API-Key": config.RAG_API_KEY}, timeout=120)
    except requests.RequestException as e:
        raise ToolError(f"RAG API unreachable: {type(e).__name__}")
    if r.status_code in (401, 403):
        raise ToolConfigError(f"RAG API rejected the key (HTTP {r.status_code})")
    if r.status_code >= 400:
        raise ToolError(f"RAG API HTTP {r.status_code}")
    data = r.json()
    # passages are dropped on purpose: they cost context on a 4B model
    return json.dumps({"answer": data.get("answer", ""), "sources": data.get("sources", [])},
                      ensure_ascii=False)
