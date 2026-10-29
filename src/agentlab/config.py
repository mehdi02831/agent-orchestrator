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
RAG_API_KEY = os.getenv("RAG_API_KEY", "")
FETCH_ALLOWED_HOSTS = {h.strip() for h in os.getenv("FETCH_ALLOWED_HOSTS", "docs.djangoproject.com").split(",") if h.strip()}
MAX_TOOL_OUTPUT_CHARS = 4000

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "127.0.0.1")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_USER = os.getenv("POSTGRES_USER", "agent")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
POSTGRES_DB = os.getenv("POSTGRES_DB", "agentlab")


def postgres_conn(dbname=None):
    return {"host": POSTGRES_HOST, "port": POSTGRES_PORT, "user": POSTGRES_USER,
            "password": POSTGRES_PASSWORD, "dbname": dbname or POSTGRES_DB, "connect_timeout": 5}

CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))

WRITE_DIR = WORKSPACE / "out"
MAX_WRITE_CHARS = 20000
WRITE_SUFFIXES = {".md", ".txt", ".csv", ".json"}
