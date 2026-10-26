from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
RAG_API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")
WORKSPACE = ROOT / "workspace"
REFUSAL_TEXT = "I don't know based on the documentation."