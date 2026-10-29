import pytest

from agentlab import config
from agentlab.tools import registry
from agentlab.tools.base import ToolError
from agentlab.tools.write import write_file


@pytest.fixture(autouse=True)
def out_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WRITE_DIR", tmp_path)
    return tmp_path


def test_write_ok(out_dir):
    msg = write_file("summary.md", "hello")
    assert (out_dir / "summary.md").read_text(encoding="utf-8") == "hello" and "5 chars" in msg


@pytest.mark.parametrize("p", ["workspace/out/summary.md", "out/summary.md", "workspace\\out\\summary.md"])
def test_prefixes_normalised(out_dir, p):
    write_file(p, "x")
    assert (out_dir / "summary.md").exists()


@pytest.mark.parametrize("p", ["../x.md", "..\\x.md", "out/../../x.md", "C:\\Windows\\x.md", "/etc/x.md"])
def test_escape_blocked(out_dir, p):
    with pytest.raises(ToolError):
        write_file(p, "x")
    assert not list(out_dir.iterdir())


@pytest.mark.parametrize("p", ["a.py", "a.exe", "noext"])
def test_suffix_blocked(p):
    with pytest.raises(ToolError, match="not allowed"):
        write_file(p, "x")


@pytest.mark.parametrize("p", ["", "   ", "docs/"])
def test_empty_or_directory_path(p):
    with pytest.raises(ToolError):
        write_file(p, "x")


def test_overwrite_needs_flag(out_dir):
    write_file("a.md", "one")
    with pytest.raises(ToolError, match="overwrite"):
        write_file("a.md", "two")
    write_file("a.md", "two", overwrite=True)
    assert (out_dir / "a.md").read_text(encoding="utf-8") == "two"


def test_content_limits(monkeypatch):
    monkeypatch.setattr(config, "MAX_WRITE_CHARS", 10)
    with pytest.raises(ToolError, match="too large"):
        write_file("a.md", "x" * 11)
    with pytest.raises(ToolError, match="string"):
        write_file("a.md", 5)


def test_subfolder_created(out_dir):
    write_file("notes/day1.md", "x")
    assert (out_dir / "notes" / "day1.md").exists()


def test_not_an_agent_tool():
    assert "write_file" not in registry.TOOLS
