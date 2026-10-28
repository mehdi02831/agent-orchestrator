from urllib.parse import urljoin, urlparse
import requests
from .. import config
from .base import ToolError

MAX_REDIRECTS = 3


def _check(url):
    parts = urlparse(url)
    if parts.scheme != "https":
        raise ToolError("only https URLs are allowed")
    if parts.hostname not in config.FETCH_ALLOWED_HOSTS:
        raise ToolError(f"host not in allow-list: {parts.hostname}")


def web_fetch(url: str) -> str:
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        _check(current)
        try:
            r = requests.get(current, timeout=10, allow_redirects=False)
        except requests.RequestException as e:
            raise ToolError(f"fetch failed: {type(e).__name__}")
        if 300 <= r.status_code < 400:
            loc = r.headers.get("Location")
            if not loc:
                raise ToolError(f"redirect without Location (HTTP {r.status_code})")
            current = urljoin(current, loc)
            continue
        if r.status_code >= 400:
            raise ToolError(f"HTTP {r.status_code}")
        if not r.text.strip():
            raise ToolError("empty response")
        return r.text[:20000]
    raise ToolError("too many redirects")
