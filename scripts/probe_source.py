import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentlab.agents.hitl_run import SOURCE_TASK
from agentlab.agents.specialist import run_specialist
from agentlab.llm.factory import get_provider
from agentlab.tools.rag import rag_ask


def safe(s):
    return str(s).encode("ascii", "replace").decode()


prompt = "Write a summary of the Django ORM docs to workspace/out/summary.md."
out = run_specialist(get_provider(), "rag", SOURCE_TASK.format(prompt=prompt))
print("SPECIALIST tool calls:")
for c in out.tool_calls:
    print(" ", c["name"], "[" + c["kind"] + "]", safe(json.dumps(c["arguments"])), "->", safe(c["content"][:150]))
print("SPECIALIST text:", safe(out.text[:200]))
for q in ("What does the Django documentation say about the ORM?", "How do QuerySets work in Django?"):
    print("DIRECT:", q, "->", safe(rag_ask(q)[:150]))
