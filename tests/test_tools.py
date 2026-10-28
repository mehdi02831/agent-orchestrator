import json
import pytest
from agentlab import config
from agentlab.tools import rag, web
from agentlab.tools.base import ToolError, ToolConfigError
from agentlab.tools.calculator import calculator
from agentlab.tools.files import read_file
from agentlab.tools.registry import execute


# calculator
@pytest.mark.parametrize("expr,expected", [
    ("0.175 * 2480", "434.0"), ("(1234 * 56) + 789", "69893"),
    ("2 ** 12 / 8", "512.0"), ("80 * 0.85 * 1.2", "81.6"), ("-3 + 5", "2"),
])
def test_calculator_ok(expr, expected):
    assert calculator(expr) == expected


@pytest.mark.parametrize("expr", [
    "__import__('os').system('dir')", "open('x')", "1 / 0", "2 ** 5000", "abs(-1)", "1 +", "a + 1",
])
def test_calculator_rejects(expr):
    with pytest.raises(ToolError):
        calculator(expr)


# read_file sandbox
def test_read_file_ok():
    assert "widget" in read_file("fixtures/prices.csv")


@pytest.mark.parametrize("p", ["../.env", "..\\.env", "fixtures/../../.env", "C:\\Windows\\win.ini", "/etc/passwd"])
def test_read_file_blocks_escape(p):
    with pytest.raises(ToolError):
        read_file(p)


def test_read_file_missing():
    with pytest.raises(ToolError, match="not found"):
        read_file("fixtures/missing.csv")


# web_fetch allow-list (no network is reached)
@pytest.mark.parametrize("url", [
    "http://docs.djangoproject.com/", "https://docs.invalid.example/data.json", "ftp://x", "not a url",
])
def test_web_fetch_blocked(url):
    with pytest.raises(ToolError):
        web.web_fetch(url)


# rag_ask
def test_rag_missing_key_is_config_error(monkeypatch):
    monkeypatch.setattr(config, "RAG_API_KEY", "")
    with pytest.raises(ToolConfigError):
        rag.rag_ask("q")


class _Resp:
    def __init__(self, code, body):
        self.status_code, self._b = code, body

    def json(self):
        return self._b


def test_rag_401_is_config_error(monkeypatch):
    monkeypatch.setattr(config, "RAG_API_KEY", "k")
    monkeypatch.setattr(rag.requests, "post", lambda *a, **k: _Resp(401, {}))
    with pytest.raises(ToolConfigError):
        rag.rag_ask("q")


def test_rag_drops_passages_and_keeps_refusal_exact(monkeypatch):
    monkeypatch.setattr(config, "RAG_API_KEY", "k")
    body = {"answer": config.REFUSAL_TEXT, "sources": [], "passages": []}
    monkeypatch.setattr(rag.requests, "post", lambda *a, **k: _Resp(200, body))
    out = json.loads(rag.rag_ask("q"))
    assert out == {"answer": config.REFUSAL_TEXT, "sources": []}


# registry classification
def test_registry_unknown_tool():
    assert execute("nope", {}).kind == "invalid_call"


def test_registry_bad_args():
    assert execute("calculator", {}).kind == "invalid_call"
    assert execute("calculator", {"expression": 5}).kind == "invalid_call"
    assert execute("calculator", {"expression": "1", "x": "y"}).kind == "invalid_call"


def test_registry_tool_error_vs_ok():
    assert execute("read_file", {"path": "fixtures/missing.csv"}).kind == "tool_error"
    r = execute("calculator", {"expression": "6 * 7"})
    assert r.ok and r.content == "42"


class _Page:
    def __init__(self, code, text):
        self.status_code, self.text = code, text


@pytest.mark.parametrize("code", [301, 302, 307])
def test_web_fetch_redirect_is_error(monkeypatch, code):
    monkeypatch.setattr(config, "FETCH_ALLOWED_HOSTS", {"docs.djangoproject.com"})
    monkeypatch.setattr(web.requests, "get", lambda *a, **k: _Page(code, ""))
    with pytest.raises(ToolError, match="redirect"):
        web.web_fetch("https://docs.djangoproject.com/x")


def test_web_fetch_empty_is_error(monkeypatch):
    monkeypatch.setattr(config, "FETCH_ALLOWED_HOSTS", {"docs.djangoproject.com"})
    monkeypatch.setattr(web.requests, "get", lambda *a, **k: _Page(200, "   "))
    with pytest.raises(ToolError, match="empty"):
        web.web_fetch("https://docs.djangoproject.com/x")


def test_web_fetch_ok(monkeypatch):
    monkeypatch.setattr(config, "FETCH_ALLOWED_HOSTS", {"docs.djangoproject.com"})
    monkeypatch.setattr(web.requests, "get", lambda *a, **k: _Page(200, "hello"))
    assert web.web_fetch("https://docs.djangoproject.com/x") == "hello"
