from .base import LLMProvider
from .. import config


def get_provider() -> LLMProvider:
    if config.LLM_PROVIDER == "ollama":
        from .ollama import OllamaProvider
        return OllamaProvider(config.OLLAMA_BASE_URL, config.OLLAMA_MODEL)
    raise ValueError(f"Unknown or disabled provider: {config.LLM_PROVIDER}")