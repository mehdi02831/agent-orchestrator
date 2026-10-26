import time
import requests
from .base import LLMProvider, LLMResponse, ToolCall


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str, model: str, timeout: int = 120):
        self.base_url, self.model, self.timeout = base_url.rstrip("/"), model, timeout

    def chat(self, messages, tools=None) -> LLMResponse:
        payload = {"model": self.model, "messages": messages,
                   "stream": False, "think": False}
        if tools:
            payload["tools"] = tools
        t0 = time.perf_counter()
        r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        msg = data.get("message", {})
        calls = [ToolCall(c["function"]["name"], c["function"].get("arguments", {}))
                 for c in msg.get("tool_calls", [])]
        return LLMResponse(msg.get("content", ""), calls, time.perf_counter() - t0, data)