import requests

TERMINAL = {"SUCCESS", "FAILURE"}


class ApiError(Exception):
    def __init__(self, status, detail):
        super().__init__(f"{status}: {detail}")
        self.status, self.detail = status, detail


def is_terminal(state: str) -> bool:
    return state in TERMINAL


class ApiClient:
    def __init__(self, base_url, api_key, session=None, timeout=30):
        self.base, self.key = base_url.rstrip("/"), api_key
        self.session, self.timeout = session or requests.Session(), timeout

    def _call(self, method, path, auth=True, **kw):
        headers = {"X-API-Key": self.key} if auth else {}
        try:
            r = self.session.request(method, self.base + path, headers=headers, timeout=self.timeout, **kw)
        except requests.RequestException as e:
            raise ApiError(0, f"API unreachable ({type(e).__name__})")
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except (ValueError, AttributeError):
                detail = r.text
            raise ApiError(r.status_code, str(detail)[:300])
        return r.json()

    def health(self):
        return self._call("GET", "/health", auth=False)

    def submit(self, prompt):
        return self._call("POST", "/tasks", json={"prompt": prompt})["task_id"]

    def status(self, task_id):
        return self._call("GET", f"/tasks/{task_id}")

    def pending(self):
        return self._call("GET", "/pending")

    def decide(self, pending_id, decision, edits=None, note=""):
        return self._call("POST", f"/pending/{pending_id}/decide",
                          json={"decision": decision, "edits": edits or None, "note": note})
