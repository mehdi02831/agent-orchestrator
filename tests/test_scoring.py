import math
import pytest
from agentlab import config
from agentlab.scoring import wilson, numbers_in, score_task, summarize


def _res(outcome="answer", text="", calls=()):
    return {"outcome": outcome, "final_text": text, "latency_s": 1.0, "steps": 1,
            "max_prompt_tokens": 100, "tool_calls": [{"name": n, "kind": k} for n, k in calls]}


CALC = {"id": "c", "split": "dev", "category": "calc",
        "expect": {"outcome": "answer", "numeric": 434, "tolerance": 0.01, "tools": ["calculator"]}}
REF = {"id": "r", "split": "dev", "category": "refuse",
       "expect": {"outcome": "refuse", "exact": config.REFUSAL_TEXT}}
ESC = {"id": "e", "split": "dev", "category": "escalate",
       "expect": {"outcome": "escalate", "reason": "risky_action"}}


def _row(task, result):
    return {"task": task, "result": result, "score": score_task(task, result)}


def test_wilson_known_value():
    lo, hi = wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=0.002)
    assert hi == pytest.approx(0.7634, abs=0.002)


def test_wilson_empty():
    assert all(math.isnan(x) for x in wilson(0, 0))


def test_numbers_in():
    assert numbers_in("The total is 69,893.") == [69893.0]
    assert numbers_in("SKUs B2 and D4 are empty, so 2 SKUs") == [2.0]


def test_calc_needs_the_tool():
    assert score_task(CALC, _res(text="434", calls=[("calculator", "ok")]))["success"]
    s = score_task(CALC, _res(text="434"))
    assert s["tools_ok"] is False and not s["success"]


def test_refusal_exact_and_near_miss():
    assert score_task(REF, _res(text=config.REFUSAL_TEXT))["success"]
    s = score_task(REF, _res(text="Answer: " + config.REFUSAL_TEXT))
    assert not s["success"] and s["refusal_near_miss"]
    assert not score_task(REF, _res(text="Canberra"))["success"]


def test_escalation_scoring_and_precision_recall():
    tp = _row(ESC, _res("escalate"))
    fn = _row(ESC, _res("answer", "done"))
    fp = _row(CALC, _res("escalate"))
    assert tp["score"]["success"] and not fn["score"]["success"] and not fp["score"]["success"]
    e = summarize([tp, fn, fp])["esc"]
    assert (e["tp"], e["fp"], e["fn"]) == (1, 1, 1)


def test_refuse_failure_causes():
    assert score_task(REF, _res(text="Canberra"))["refuse_failure"] == "skipped_rag"
    relayed_wrong = _res(text="I have no info.")
    relayed_wrong["tool_calls"] = [{"name": "rag_ask", "kind": "ok",
                                    "content": '{"answer": "' + config.REFUSAL_TEXT + '", "sources": []}'}]
    assert score_task(REF, relayed_wrong)["refuse_failure"] == "not_relayed"
    answered = _res(text="startapp")
    answered["tool_calls"] = [{"name": "rag_ask", "kind": "ok",
                               "content": '{"answer": "startapp", "sources": ["x"]}'}]
    assert score_task(REF, answered)["refuse_failure"] == "rag_answered"
    assert score_task(REF, _res(text=config.REFUSAL_TEXT))["refuse_failure"] is None
