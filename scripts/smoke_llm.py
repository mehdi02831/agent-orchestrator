import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentlab.llm.factory import get_provider

r = get_provider().chat([{"role": "user", "content": "Reply with exactly: pong"}])
print(repr(r.content), f"{r.latency_s:.2f}s")