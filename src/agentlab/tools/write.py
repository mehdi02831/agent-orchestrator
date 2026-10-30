from .. import config
from .base import ToolError


def _normalize(path: str) -> str:
    p = path.strip().replace("\\", "/")
    for prefix in ("workspace/out/", "out/"):
        if p.startswith(prefix):
            return p[len(prefix):]
    if p.startswith("workspace/"):
        raise ToolError("path must be under workspace/out/")
    return p


def resolve_write_path(path):
    if not isinstance(path, str) or not path.strip():
        raise ToolError("empty path")
    rel = _normalize(path)
    if not rel or rel.endswith("/"):
        raise ToolError("path must name a file")
    base = config.WRITE_DIR.resolve()
    try:
        target = (base / rel).resolve()
    except (OSError, ValueError) as e:
        raise ToolError(f"invalid path: {e}")
    if target == base or not target.is_relative_to(base):
        raise ToolError("path is outside the writable folder")
    if target.suffix.lower() not in config.WRITE_SUFFIXES:
        raise ToolError(f"file type not allowed: {target.suffix or '(none)'}")
    return target


def validate_write(path, content):
    if not isinstance(content, str):
        raise ToolError("content must be a string")
    if len(content) > config.MAX_WRITE_CHARS:
        raise ToolError("content too large")
    return resolve_write_path(path)


def write_file(path, content, overwrite=False):
    """NOT registered as an agent tool. Only called after a human approves."""
    target = validate_write(path, content)
    if target.exists() and not overwrite:
        raise ToolError("file exists; overwrite was not approved")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {target.name}"
