import json
from pathlib import Path

from agentlab.agents.supervisor import run_supervisor
from agentlab.llm.base import LLMResponse
from agentlab.memory.recall import build_note


def hit(sim, prompt="Q?", answer="A!"):
    return {"run_id": 1, "prompt": prompt, "answer": answer, "similarity": sim}


def test_note_empty_without_hits():
    assert build_note([], 0.5) == ""


def test_note_empty_below_threshold():
    assert build_note([hit(0.49)], 0.5) == ""


def test_note_uses_best_hit_and_states_uncertainty():
    note = build_note([hit(0.6, "low", "no"), hit(0.9, "How do I make an app?", "startapp")], 0.5)
    assert "How do I make an app?" in note and "startapp" in note and "low" not in note
    assert "may be wrong or outdated" in note


class Rec:
    def __init__(self, sup, spec=()):
        self.sup, self.spec, self.sup_msgs, self.spec_msgs = list(sup), list(spec), [], []

    def chat(self, messages, tools=None, response_format=None):
        if response_format is not None:
            self.sup_msgs.append(messages)
            return self.sup.pop(0)
        self.spec_msgs.append(messages)
        return self.spec.pop(0)


def _finish():
    return LLMResponse(json.dumps({"action": "finish", "answer": "ok"}))


def test_supervisor_receives_note_and_specialists_do_not():
    llm = Rec(sup=[LLMResponse(json.dumps({"action": "delegate", "agent": "data", "task": "do it"})), _finish()],
              spec=[LLMResponse("done")])
    run_supervisor(llm, "What is 6 * 7?", note="NOTE-X")
    assert all("NOTE-X" in m[1]["content"] for m in llm.sup_msgs)
    assert "NOTE-X" not in json.dumps(llm.spec_msgs)


def test_supervisor_default_has_no_note():
    llm = Rec(sup=[_finish()])
    run_supervisor(llm, "What is 6 * 7?")
    assert llm.sup_msgs[0][1]["content"] == "What is 6 * 7?"


def test_frozen_runner_never_imports_memory():
    src = (Path(__file__).resolve().parents[1] / "eval" / "run_eval.py").read_text(encoding="utf-8")
    assert "agentlab.memory" not in src
