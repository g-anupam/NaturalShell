# ============================================================
#  history_rag.py
#  Parses ~/.zsh_history -> embeds into ChromaDB -> semantic search
# ============================================================

from __future__ import annotations
import re, os, hashlib, sys, warnings, logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict

# Suppress HuggingFace / tokenizer / BertModel noise before any imports
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

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

    # Suppress the BertModel LOAD REPORT printed to stdout
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
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

        docs.append(f"{entry['command']}  [run on {ts_human}]")
        ids.append(eid)
        metas.append({
            "command":  entry["command"],
            "ts_iso":   ts_iso,
            "ts_human": ts_human,
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


# ── Search ────────────────────────────────────────────────────

def search_history(query: str, top_k: int = None) -> List[Dict]:
    col = _collection()
    k   = min(top_k or config.RAG_TOP_K, max(col.count(), 1))

    if col.count() == 0:
        return []

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
    return hits
