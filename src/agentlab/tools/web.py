from urllib.parse import urlparse
import requests
from .. import config
from .base import ToolError


def web_fetch(url: str) -> str:
    parts = urlparse(url)
    if parts.scheme != "https":
        raise ToolError("only https URLs are allowed")
    if parts.hostname not in config.FETCH_ALLOWED_HOSTS:
        raise ToolError(f"host not in allow-list: {parts.hostname}")
    try:
        r = requests.get(url, timeout=10, allow_redirects=False)
    except requests.RequestException as e:
        raise ToolError(f"fetch failed: {type(e).__name__}")
    if r.status_code >= 400:
        raise ToolError(f"HTTP {r.status_code}")
    return r.text[:20000]
