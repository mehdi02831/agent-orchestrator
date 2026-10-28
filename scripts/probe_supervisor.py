import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentlab.agents.supervisor import run_supervisor
from agentlab.llm.factory import get_provider

PROMPTS = ["What is 17.5% of 2480?",
           "According to the Django docs, which command creates a new app? Also, what is 6 * 7?",
           "What is the capital of Australia?",
           "Fix it."]
p = get_provider()
for q in PROMPTS:
    r = run_supervisor(p, q)
    print("=" * 60)
    print("Q:", q)
    print("outcome:", r.outcome, "| steps:", r.steps, f"| {r.latency_s:.1f}s | invalid_plans:", r.invalid_plans)
    print("delegations:", [(d["agent"], d["task"][:60]) for d in r.delegations])
    print("tools:", [c["name"] for c in r.tool_calls])
    print("reason_type:", r.reason_type, "| error:", r.error)
    print("final:", (r.final_text or r.escalation_reason)[:160].encode("ascii", "replace").decode())
