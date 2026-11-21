"""Knowledge Base retrieval over the _KB Markdown articles.

Small, dependency-free BM25 index (title/tags weighted) - good enough for ~80
short single-topic policy articles, and swappable for a vector store behind the
same `search()` contract. Read-only; returns excerpts + policy IDs for citation.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.core.config import settings

_TOKEN = re.compile(r"[a-z0-9$]+")
_STOP = {"the", "a", "an", "of", "to", "and", "or", "is", "are", "my", "i", "for", "in", "on",
         "it", "can", "do", "how", "what", "me", "you", "your", "be", "if", "with", "that",
         "policy", "policies", "have", "long", "something", "get", "want", "need",
         "please", "about", "tell", "there", "was", "will", "did", "does"}   # every article is a policy; the word carries no signal


def _stem(t: str) -> str:
    return t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t


def _tokens(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass(frozen=True)
class Article:
    policy_id: str
    title: str
    path: str  # relative to _KB, posix
    tags: tuple[str, ...]
    body: str
    tf: Counter
    length: int
    fields: tuple[frozenset, frozenset, frozenset]   # title, folder/filename, tag tokens


def _parse(path: Path, root: Path) -> Article | None:
    raw = path.read_text(encoding="utf-8")
    meta: dict[str, str] = {}
    body = raw
    if raw.startswith("---"):
        _, fm, body = raw.split("---", 2)
        for line in fm.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip().strip('"')
    pid = meta.get("policy_id")
    if not pid:
        return None
    title = meta.get("title", path.stem)
    tags = tuple(t.strip() for t in meta.get("tags", "").strip("[]").split(",") if t.strip())
    folder = path.parent.name.replace("-", " ") + " " + path.stem.replace("-", " ")
    weighted = (title + " ") * 3 + (folder + " ") * 3 + (" ".join(tags) + " ") * 2 + body
    toks = _tokens(weighted)
    return Article(pid, title, path.relative_to(root).as_posix(), tags, body.strip(),
                   Counter(toks), len(toks),
                   (frozenset(_tokens(title)), frozenset(_tokens(folder)), frozenset(_tokens(" ".join(tags)))))


@lru_cache(maxsize=1)
def _index() -> tuple[list[Article], dict[str, int], float]:
    root = settings.kb_dir
    arts = [a for p in sorted(root.rglob("*.md"))
            if not p.name.startswith("README") and "_eval" not in p.parts
            for a in [_parse(p, root)] if a]
    df: Counter = Counter()
    for a in arts:
        df.update(a.tf.keys())
    avg = sum(a.length for a in arts) / max(len(arts), 1)
    return arts, dict(df), avg


def article_count() -> int:
    return len(_index()[0])


def _excerpt(body: str, limit: int = 1400) -> str:
    text = re.sub(r"\n{3,}", "\n\n", body).strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


def search(query: str, top_k: int | None = None) -> list[dict]:
    """BM25 search. Returns [{policy_id,title,path,score,excerpt}] best-first."""
    arts, df, avg = _index()
    q = _tokens(query)
    if not q or not arts:
        return []
    k1, b, n = 1.5, 0.75, len(arts)
    scored = []
    for a in arts:
        s = 0.0
        for t in set(q):
            f = a.tf.get(t, 0)
            if not f:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * a.length / avg))
        if s > 0:
            # Field boosts: a query word in the title/folder/tags is a much stronger signal than in the body.
            qs = set(q)
            s += 2.0 * len(qs & a.fields[0]) + 1.5 * len(qs & a.fields[1]) + 1.0 * len(qs & a.fields[2])
            if a.policy_id.endswith("-001"):
                s += 0.5      # each folder's -001 article is its canonical overview
            scored.append((s, a))
    scored.sort(key=lambda x: -x[0])
    return [{"policy_id": a.policy_id, "title": a.title, "path": a.path,
             "score": round(s, 2), "excerpt": _excerpt(a.body)}
            for s, a in scored[: top_k or settings.agent_kb_top_k]]
