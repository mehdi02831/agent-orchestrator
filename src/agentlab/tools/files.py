from .. import config
from .base import ToolError

MAX_BYTES = 200_000


def read_file(path: str) -> str:
    base = config.WORKSPACE.resolve()
    try:
        target = (base / path).resolve()
    except (OSError, ValueError) as e:
        raise ToolError(f"invalid path: {e}")
    if not target.is_relative_to(base):
        raise ToolError("path is outside the workspace")
    if not target.is_file():
        raise ToolError(f"file not found: {path}")
    if target.stat().st_size > MAX_BYTES:
        raise ToolError("file too large")
    return target.read_text(encoding="utf-8", errors="replace")
