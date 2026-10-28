from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    latency_s: float = 0.0
    raw: dict = field(default_factory=dict)


class LLMProvider(ABC):
    """One swappable class. Messages use the OpenAI-style role/content dicts.
    response_format: optional JSON schema that constrains the reply (not combined with tools)."""

    @abstractmethod
    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             response_format: dict | None = None) -> LLMResponse:
        ...
