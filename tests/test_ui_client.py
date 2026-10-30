import pytest
import requests

from agentlab.ui.client import ApiClient, ApiError, is_terminal


class Resp:
    def __init__(self, code=200, body=None, text=""):
        self.status_code, self._body, self.text = code, body, text

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class Sess:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.calls = resp, exc, []

    def request(self, method, url, headers=None, timeout=None, **kw):
        self.calls.append({"method": method, "url": url, "headers": headers, **kw})
        if self.exc:
            raise self.exc
        return self.resp


def client(sess):
    return ApiClient("http://api:8000/", "k", session=sess)


def test_sends_key_and_url():
    s = Sess(Resp(body={"pending": []}))
    assert client(s).pending() == {"pending": []}
    assert s.calls[0]["url"] == "http://api:8000/pending" and s.calls[0]["headers"] == {"X-API-Key": "k"}


def test_submit_returns_task_id():
    s = Sess(Resp(body={"task_id": "t1"}))
    assert client(s).submit("hi") == "t1"
    assert s.calls[0]["method"] == "POST" and s.calls[0]["json"] == {"prompt": "hi"}


def test_http_error_becomes_apierror():
    with pytest.raises(ApiError) as e:
        client(Sess(Resp(401, {"detail": "Invalid or missing API key"}))).pending()
    assert e.value.status == 401 and "Invalid" in e.value.detail


def test_validation_detail_list_is_stringified():
    with pytest.raises(ApiError) as e:
        client(Sess(Resp(422, {"detail": [{"msg": "too short"}]}))).submit("")
    assert e.value.status == 422 and "too short" in e.value.detail


def test_network_error():
    with pytest.raises(ApiError) as e:
        client(Sess(exc=requests.ConnectionError("down"))).health()
    assert e.value.status == 0 and "ConnectionError" in e.value.detail


def test_non_json_error_body():
    with pytest.raises(ApiError) as e:
        client(Sess(Resp(502, None, text="Bad gateway"))).status("x")
    assert e.value.status == 502 and e.value.detail == "Bad gateway"


def test_decide_payload():
    s = Sess(Resp(body={"status": "approved", "message": "ok"}))
    client(s).decide(5, "edit", {"content": "x"}, "n")
    assert s.calls[0]["json"] == {"decision": "edit", "edits": {"content": "x"}, "note": "n"}
    client(s).decide(5, "reject")
    assert s.calls[1]["json"]["edits"] is None


def test_health_sends_no_key():
    s = Sess(Resp(body={"status": "ok"}))
    client(s).health()
    assert s.calls[0]["headers"] == {}


@pytest.mark.parametrize("state,done", [("SUCCESS", True), ("FAILURE", True), ("PENDING", False), ("STARTED", False)])
def test_is_terminal(state, done):
    assert is_terminal(state) is done
