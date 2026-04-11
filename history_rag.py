# ============================================================
#  history_rag.py
#  Three retrieval modes:
#    1. RECENT   - sort by timestamp, return last N
#    2. SEMANTIC - ChromaDB vector similarity
#    3. HYBRID   - semantic search filtered by time window
# ============================================================

from __future__ import annotations
import re, os, hashlib, sys, warnings, logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import chromadb
from chromadb.utils import embedding_functions

sys.path.insert(0, str(Path(__file__).parent))
import config


# ── Parse ─────────────────────────────────────────────────────

def parse_zsh_history() -> List[Dict]:
    """
    Returns list of { command, timestamp } dicts in file order
    (last = most recent). Handles extended and plain zsh formats.
    """
    path = config.ZSH_HISTORY_PATH
    if not os.path.exists(path):
        print(f"[warn] zsh history not found at {path}")
        return []

    extended_re = re.compile(r"^: (\d+):\d+;(.+)$")
    entries = []

    with open(path, "r", errors="replace") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        while line.endswith("\\") and i + 1 < len(lines):
            i += 1
            line = line[:-1] + " " + lines[i].rstrip("\n")

        m = extended_re.match(line)
        if m:
            ts  = datetime.fromtimestamp(int(m.group(1)))
            cmd = m.group(2).strip()
        else:
            ts  = None
            cmd = line.strip()

        if cmd:
            entries.append({"command": cmd, "timestamp": ts})
        i += 1

    return entries


# ── ChromaDB ──────────────────────────────────────────────────

def _collection():
    os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL
        )
    return client.get_or_create_collection(
        name="zsh_history",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def _make_id(index: int, entry: Dict) -> str:
    key = f"{index}|{entry['command']}|{entry['timestamp']}"
    return hashlib.md5(key.encode()).hexdigest()


# ── Index ─────────────────────────────────────────────────────

def index_history(force: bool = False) -> int:
    entries = parse_zsh_history()
    if not entries:
        return 0

    col      = _collection()
    existing = set(col.get()["ids"])
    docs, ids, metas = [], [], []
    total_added = 0

    for idx, entry in enumerate(entries):
        eid = _make_id(idx, entry)
        if eid in existing and not force:
            continue

        ts_iso   = entry["timestamp"].isoformat() if entry["timestamp"] else "unknown"
        ts_human = entry["timestamp"].strftime("%Y-%m-%d %H:%M") if entry["timestamp"] else "unknown date"
        # Store unix timestamp as string for filtering
        ts_unix  = str(int(entry["timestamp"].timestamp())) if entry["timestamp"] else "0"

        docs.append(f"{entry['command']}  [run on {ts_human}]")
        ids.append(eid)
        metas.append({
            "command":  entry["command"],
            "ts_iso":   ts_iso,
            "ts_human": ts_human,
            "ts_unix":  ts_unix,
        })

        if len(docs) >= config.HISTORY_CHUNK_SIZE:
            col.upsert(documents=docs, ids=ids, metadatas=metas)
            total_added += len(docs)
            docs, ids, metas = [], [], []

    if docs:
        col.upsert(documents=docs, ids=ids, metadatas=metas)
        total_added += len(docs)

    return total_added


def history_count() -> int:
    try:
        return _collection().count()
    except Exception:
        return 0


# ── Retrieval modes ───────────────────────────────────────────

def _fmt(hits: List[Dict]) -> str:
    if not hits:
        return "No relevant past commands found in history."
    lines = [
        f"Found {len(hits)} command(s) from the user's actual zsh history.",
        f"LIST ALL {len(hits)} of them in your answer. Do not skip any.\n",
    ]
    for i, h in enumerate(hits, 1):
        lines.append(f"  {i}. `{h['command']}`")
        lines.append(f"     when: {h['when']}")
        if "relevance" in h:
            lines.append(f"     relevance: {h['relevance']}%")
    return "\n".join(lines)


def get_recent(n: int = 5) -> str:
    """
    Returns the N most recent commands by timestamp.
    Used for queries like 'last 2 commands', 'what did i just run'.
    Falls back to file-order for entries with no timestamp.
    """
    entries = parse_zsh_history()
    if not entries:
        return "No history found."

    # Sort: entries with timestamps first by ts desc, then no-ts entries by position desc
    with_ts    = [(e, e["timestamp"]) for e in entries if e["timestamp"]]
    without_ts = [e for e in entries if not e["timestamp"]]

    with_ts.sort(key=lambda x: x[1], reverse=True)
    recent_ts  = [e for e, _ in with_ts[:n]]

    # If we don't have enough timestamped entries, pad from end of file
    if len(recent_ts) < n:
        recent_ts += without_ts[-(n - len(recent_ts)):]

    hits = []
    for e in recent_ts[:n]:
        ts_human = e["timestamp"].strftime("%Y-%m-%d %H:%M") if e["timestamp"] else "unknown date"
        hits.append({"command": e["command"], "when": ts_human})

    return _fmt(hits)


def search_semantic(query: str, top_k: int = None) -> str:
    """
    Pure semantic similarity search via ChromaDB.
    Used for queries like 'that ffmpeg command i used to compress'.
    """
    col = _collection()
    k   = min(top_k or config.RAG_TOP_K, max(col.count(), 1))
    if col.count() == 0:
        return "History not indexed yet. Run: agent --reindex"

    results = col.query(
        query_texts=[query],
        n_results=k,
        include=["metadatas", "distances"],
    )

    hits = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        hits.append({
            "command":   meta["command"],
            "when":      meta["ts_human"],
            "relevance": round((1 - dist) * 100, 1),
        })
    return _fmt(hits)


def search_hybrid(query: str, days: int = None, top_k: int = None) -> str:
    """
    Semantic search scoped to a time window.
    Used for queries like 'pip command i ran last week'.
    days=None means no time filter (falls back to pure semantic).
    """
    col = _collection()
    if col.count() == 0:
        return "History not indexed yet. Run: agent --reindex"

    k = min((top_k or config.RAG_TOP_K) * 3, col.count())

    # Fetch more candidates then filter by time
    results = col.query(
        query_texts=[query],
        n_results=k,
        include=["metadatas", "distances"],
    )

    hits = []
    cutoff_unix = int((datetime.now() - timedelta(days=days)).timestamp()) if days else 0

    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        ts_unix = int(meta.get("ts_unix", "0") or "0")
        if days and ts_unix and ts_unix < cutoff_unix:
            continue
        hits.append({
            "command":   meta["command"],
            "when":      meta["ts_human"],
            "relevance": round((1 - dist) * 100, 1),
        })

    # Trim to top_k after filtering
    hits = hits[:top_k or config.RAG_TOP_K]
    return _fmt(hits)
