import shutil, socket, sys
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentlab import config


def port_open(port, host="127.0.0.1"):
    with socket.socket() as s:
        s.settimeout(1.5)
        return s.connect_ex((host, port)) == 0


def http_ok(url):
    try:
        return requests.get(url, timeout=3).status_code < 500
    except Exception:
        return False


checks = {
    "git on PATH": shutil.which("git") is not None,
    "Ollama /api/tags": http_ok(f"{config.OLLAMA_BASE_URL}/api/tags"),
    "RAG API /health": http_ok(f"{config.RAG_API_URL}/health"),
    "Postgres :5432": port_open(5432),
    "Redis :6379": port_open(6379),
    "Chroma :8001 heartbeat": http_ok("http://127.0.0.1:8001/api/v2/heartbeat"),
}
for name, ok in checks.items():
    print(f"[{'OK' if ok else 'FAIL'}] {name}")
sys.exit(0 if all(checks.values()) else 1)