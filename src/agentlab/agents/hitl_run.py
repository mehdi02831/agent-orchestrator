import json
import re
from dataclasses import dataclass

from ..hitl import decide, propose
from ..llm.base import LLMProvider
from ..memory.records import run_record
from ..tools.base import ToolError
from ..tools.write import resolve_write_path
from .guardrails import _WRITE_TO, refusal_relay  # _WRITE_TO is private; imported to leave guardrails.py unedited
from .specialist import run_specialist
from .supervisor import SupervisorResult, run_supervisor

FROZEN_GUARDS = frozenset({"failures", "gate", "provenance", "relay"})
STATUS_FOR = {"approve": "approved", "edit": "edited", "reject": "rejected"}

DRAFT_SCHEMA = {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}
DRAFT_SYSTEM = ('You write the content of a document. Reply with ONE JSON object: {"content": "<the full text>"}. '
                "Use only the source material provided. Do not invent facts. Plain text or Markdown.")
SOURCE_TASK = ('Call rag_ask with exactly this question and nothing else: "{question}" '
               "Do not write any file.")

_TARGET = re.compile(r"""\b(?:to|into)\s+(?:the\s+)?(?:file\s+)?["']?([^\s"']+)""", re.I)
_PATHLIKE = re.compile(r"[\\/]|\.\w+$")


def extract_target_path(prompt):
    """Last path-like token after 'to' or 'into', or None."""
    cands = [c.rstrip(".,;:)!?") for c in _TARGET.findall(prompt)]
    cands = [c for c in cands if c and _PATHLIKE.search(c)]
    return cands[-1] if cands else None


_VERB = re.compile(r"^\s*(?:please\s+)?(?:write|save|export|append|create|draft|produce)\s+", re.I)
_ARTICLE = re.compile(r"^(?:an?|the)\s+", re.I)
_ADJ = re.compile(r"^(?:short|brief|quick|detailed|full|complete|concise)\s+", re.I)
_NOUN = re.compile(r"^(?:summary|explanation|overview|description|report|notes?|write-?up)\s+(?:of|about|on|for)\s+", re.I)
_DOCWORD = re.compile(r"\b(?:docs?|documentation)\b", re.I)
_TAIL_PREP = re.compile(r"\s*\b(?:to|into)\b(?:\s+the)?(?:\s+file)?\s*[\"']?$", re.I)


def topic_from_prompt(prompt, path):
    """Deterministic topic: the request minus verb, filler, doc words and the target path. '' if nothing is left."""
    idx = prompt.rfind(path)
    head = prompt[:idx] if idx >= 0 else prompt
    head = _TAIL_PREP.sub("", head.strip())
    head = _VERB.sub("", head)
    head = _ARTICLE.sub("", head)
    head = _ADJ.sub("", head)
    head = _NOUN.sub("", head)
    head = _ARTICLE.sub("", head)
    head = _DOCWORD.sub("", head)
    topic = re.sub(r"\s+", " ", head).strip(" .,;:!?")
    return topic if len(topic) >= 3 else ""


def source_question(topic):
    t = topic.rstrip("?")
    return f"Explain {t}." if "django" in t.lower() else f"Explain {t} in Django."


@dataclass
class ReviewOutcome:
    result: SupervisorResult
    run_id: int
    proposal: str
    pending_id: int | None = None


def _draft(provider, prompt, source):
    msgs = [{"role": "system", "content": DRAFT_SYSTEM},
            {"role": "user", "content": f"Request: {prompt}\n\nSource material:\n{source[:4000]}"}]
    resp = provider.chat(msgs, None, response_format=DRAFT_SCHEMA)
    try:
        data = json.loads(resp.content)
    except ValueError:
        return None
    content = data.get("content") if isinstance(data, dict) else None
    if not isinstance(content, str) or not content.strip():
        return None
    return content.strip()


def _maybe_propose(provider, prompt, res, run_id, pending_store):
    if res.outcome != "escalate" or res.reason_type != "risky_action":
        return "not_risky", None
    if not _WRITE_TO.search(prompt):
        return "not_write_request", None
    path = extract_target_path(prompt)
    if path is None:
        return "no_target", None
    try:
        resolve_write_path(path)
    except ToolError:
        return "invalid_target", None
    topic = topic_from_prompt(prompt, path)
    if not topic:
        return "no_topic", None
    src = run_specialist(provider, "rag", SOURCE_TASK.format(question=source_question(topic)))  # ToolConfigError propagates
    if not any(c["name"] == "rag_ask" and c["kind"] == "ok" for c in src.tool_calls):
        return "no_source", None
    if refusal_relay(src.tool_calls):
        return "refusal", None
    content = _draft(provider, prompt, src.text)
    if content is None:
        return "bad_draft", None
    try:
        pid = propose(pending_store, path, content, prompt=prompt, reason=res.escalation_reason,
                      reason_type="risky_action", run_id=run_id, state={"source": src.text[:1000]})
    except ToolError:
        return "bad_draft", None
    return "queued", pid


def run_with_review(provider: LLMProvider, prompt, run_store, pending_store, guards=FROZEN_GUARDS) -> ReviewOutcome:
    res = run_supervisor(provider, prompt, guards=frozenset(guards))
    run_id = run_store.save_run(run_record("supervisor", prompt, res, {"guards": sorted(guards), "review": True}))
    proposal, pending_id = _maybe_propose(provider, prompt, res, run_id, pending_store)
    return ReviewOutcome(res, run_id, proposal, pending_id)


def resolve(run_store, pending_store, pending_id, decision, edits=None, note=""):
    """Human decision, then update the linked run. Returns (status, message) from decide()."""
    status, msg = decide(pending_store, pending_id, decision, edits=edits, note=note)
    if status in ("approved", "rejected"):
        row = pending_store.get(pending_id)
        if row and row.get("run_id"):
            run_store.set_status(row["run_id"], STATUS_FOR[decision])
    return status, msg
