"""
Per-project long-term memory — pure Python, no external ML dependencies.
Stores facts as JSON, retrieves by cosine similarity using a hash-based embedding.
Each project (identified by CWD) gets its own memory file.
"""
import hashlib
import json
import math
import re
import uuid
from datetime import datetime
from pathlib import Path

MEMORY_DIR = Path.home() / ".astra" / "memory"
_EMBED_DIM  = 256


# ── Embedding ─────────────────────────────────────────────────────────────────

def _embed(text: str) -> list[float]:
    """
    Lightweight word-hash embedding — no ML dependencies.
    Maps each word to a bucket via hash, then L2-normalises.
    """
    words = re.findall(r'\w+', text.lower())
    vec = [0.0] * _EMBED_DIM
    for word in words:
        vec[hash(word) % _EMBED_DIM] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ── Store ─────────────────────────────────────────────────────────────────────

class MemoryStore:
    """
    JSON-backed memory store scoped to a single project (CWD).
    Each entry: {id, fact, embedding, category, timestamp}
    """

    def __init__(self, cwd: str):
        self._cwd  = cwd
        self._file = MEMORY_DIR / (hashlib.md5(cwd.encode()).hexdigest()[:16] + ".json")
        self._data: list[dict] = []
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self):
        try:
            MEMORY_DIR.mkdir(parents=True, exist_ok=True)
            if self._file.exists():
                with open(self._file, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
        except Exception:
            self._data = []

    def _save(self):
        try:
            MEMORY_DIR.mkdir(parents=True, exist_ok=True)
            with open(self._file, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except Exception:
            pass

    # ── Public API ────────────────────────────────────────────────────────────

    def save(self, fact: str, category: str = "auto") -> bool:
        """Store a memory fact."""
        fact = fact.strip()
        if not fact:
            return False
        entry = {
            "id":        str(uuid.uuid4()),
            "fact":      fact,
            "embedding": _embed(fact),
            "category":  category,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self._data.append(entry)
        self._save()
        return True

    def retrieve(self, query: str, n: int = 5) -> list[str]:
        """Return the n most relevant facts for a query."""
        if not self._data:
            return []
        q_vec = _embed(query)
        scored = [
            (_cosine(q_vec, entry["embedding"]), entry["fact"])
            for entry in self._data
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [fact for _, fact in scored[:n] if _ > 0.0]

    def list_all(self) -> list[dict]:
        """Return all memories as dicts with fact/category/timestamp."""
        return [
            {
                "fact":      e["fact"],
                "category":  e.get("category", ""),
                "timestamp": e.get("timestamp", "")[:10],
            }
            for e in self._data
        ]

    def clear(self) -> None:
        """Delete all memories for this project."""
        self._data = []
        try:
            self._file.unlink(missing_ok=True)
        except Exception:
            pass

    @property
    def ready(self) -> bool:
        return True  # always available — pure Python
