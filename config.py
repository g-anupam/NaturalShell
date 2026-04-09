# ============================================================
#  NaturalShell — config.py
#  API keys are loaded from .env — never hardcode them here.
# ============================================================

import os
from pathlib import Path

# Load .env from the project directory
def _load_env():
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        raise FileNotFoundError(
            f"\n[error] .env file not found at {env_path}\n"
            "  Run: cp .env.example .env  and fill in your API keys."
        )
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

_load_env()

# ── Active LLM ────────────────────────────────────────────────
ACTIVE_LLM = os.environ.get("ACTIVE_LLM", "groq")  # groq | gemini | ollama

# ── API Keys (from .env) ──────────────────────────────────────
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# ── Models ────────────────────────────────────────────────────
GROQ_MODEL      = "llama-3.3-70b-versatile"
GEMINI_MODEL    = "gemini-2.0-flash"
OLLAMA_MODEL    = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434"

# ── Embeddings ────────────────────────────────────────────────
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = os.path.expanduser("~/.naturalshell/chroma_db")
ZSH_HISTORY_PATH   = os.path.expanduser("~/.zsh_history")

# ── RAG ───────────────────────────────────────────────────────
RAG_TOP_K          = 5
HISTORY_CHUNK_SIZE = 200

# ── Safety ────────────────────────────────────────────────────
DANGEROUS_PATTERNS = [
    "rm -rf", "rm -r", "mkfs", "dd if=", ":(){ :|:& };:",
    "> /dev/", "chmod 777", "sudo rm", "shutdown", "reboot",
    "wipefs", "truncate", "fdisk",
]
