from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from agentlab import config
from agentlab.api import app as api_module

client = TestClient(api_module.app)


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setattr(config, "ORCH_API_KEY", "secret-key")


def test_health_is_open():
    assert client.get("/health").json() == {"status": "ok"}


def test_missing_and_wrong_key_refused():
    assert client.post("/echo", json={"text": "hi"}).status_code == 401
    assert client.post("/echo", json={"text": "hi"}, headers={"X-API-Key": "nope"}).status_code == 401


def test_unset_server_key_refuses_everything(monkeypatch):
    monkeypatch.setattr(config, "ORCH_API_KEY", "")
    assert client.post("/echo", json={"text": "hi"}, headers={"X-API-Key": ""}).status_code == 401
    assert client.post("/echo", json={"text": "hi"}).status_code == 401


def test_echo_enqueues_with_valid_key(monkeypatch):
    sent = []
    monkeypatch.setattr(api_module.jobs, "echo",
                        SimpleNamespace(delay=lambda t: sent.append(t) or SimpleNamespace(id="abc")))
    r = client.post("/echo", json={"text": "hello"}, headers={"X-API-Key": "secret-key"})
    assert r.status_code == 200 and r.json() == {"task_id": "abc"} and sent == ["hello"]


def test_echo_validates_body():
    h = {"X-API-Key": "secret-key"}
    assert client.post("/echo", json={"text": ""}, headers=h).status_code == 422
    assert client.post("/echo", json={}, headers=h).status_code == 422
