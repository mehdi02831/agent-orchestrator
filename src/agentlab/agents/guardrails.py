import json
import re

from .. import config

_PATH = r"(?:workspace\b|[\w.-]+[\\/][\w./\\-]*|[\w-]+\.(?:csv|txt|md|json|py|log|ya?ml|html|xml|ini|cfg)\b)"
_DESTRUCTIVE = re.compile(r"\b(?:delete|erase|wipe|overwrite|truncate|remove|rename)\b.*" + _PATH, re.I)
_WRITE_TO = re.compile(r"\b(?:write|save|append|export)\b.*\b(?:to|into)\s+(?:the\s+)?(?:file\s+)?" + _PATH, re.I)


def is_risky(prompt: str) -> bool:
    """Gate: destructive verb + path-like target, or write verb + 'to/into' + path-like target."""
    return bool(_DESTRUCTIVE.search(prompt) or _WRITE_TO.search(prompt))


def refusal_relay(tool_calls: list) -> bool:
    """True if every successful tool call was rag_ask and each one returned the exact refusal."""
    ok = [c for c in tool_calls if c["kind"] == "ok" and c["name"] != "ask_human"]
    if not ok or any(c["name"] != "rag_ask" for c in ok):
        return False
    for c in ok:
        try:
            if json.loads(c["content"]).get("answer", "").strip() != config.REFUSAL_TEXT:
                return False
        except (ValueError, AttributeError):
            return False
    return True


def unrecovered_failure(tool_calls: list) -> bool:
    """True if the last call to some tool was a tool_error (invalid calls and ask_human are ignored)."""
    last = {}
    for c in tool_calls:
        if c["name"] != "ask_human":
            last[c["name"]] = c["kind"]
    return any(kind == "tool_error" for kind in last.values())
