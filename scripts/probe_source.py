import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import json
from agentlab import config
from agentlab.agents.hitl_run import SOURCE_TASK, extract_target_path, source_question, topic_from_prompt
from agentlab.agents.specialist import run_specialist
from agentlab.llm.factory import get_provider
from agentlab.tools.rag import rag_ask

PROMPTS = [
    "Write a summary of the Django ORM docs to workspace/out/summary.md.",
    "Save a short summary of Django migrations to workspace/out/migrations.md",
    "Write what the startapp command does into workspace/out/startapp.txt",
    "Export an explanation of select_related to workspace/out/orm.md",
]


def safe(s):
    return str(s).encode("ascii", "replace").decode()


p, answered = get_provider(), 0
for prompt in PROMPTS:
    topic = topic_from_prompt(prompt, extract_target_path(prompt))
    out = run_specialist(p, "rag", SOURCE_TASK.format(question=source_question(topic)))
    calls = [c for c in out.tool_calls if c["name"] == "rag_ask" and c["kind"] == "ok"]
    asked = calls[0]["arguments"].get("question") if calls else None
    ans = json.loads(rag_ask(asked)).get("answer", "") if asked else ""
    ok = bool(asked) and ans.strip() != config.REFUSAL_TEXT
    answered += int(ok)
    print("=" * 60)
    print("topic:", safe(topic), "| asked:", safe(asked))
    print("sourced:", ok, "|", safe(ans[:120]))
print(f"\n{answered} of {len(PROMPTS)} write prompts produced usable source material")
