# ============================================================
#  NaturalShell — config.py
# ============================================================

import os
from pathlib import Path

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

ACTIVE_LLM = os.environ.get("ACTIVE_LLM", "groq")

GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

GROQ_MODEL      = "llama-3.3-70b-versatile"
GEMINI_MODEL    = "gemini-2.0-flash"
OLLAMA_MODEL    = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434"

EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = os.path.expanduser("~/.naturalshell/chroma_db")
ZSH_HISTORY_PATH   = os.path.expanduser("~/.zsh_history")

RAG_TOP_K          = 5
HISTORY_CHUNK_SIZE = 200

DANGEROUS_PATTERNS = [
    "rm -rf", "rm -r", "mkfs", "dd if=", ":(){ :|:& };:",
    "> /dev/", "chmod 777", "sudo rm", "shutdown", "reboot",
    "wipefs", "truncate", "fdisk",
]
