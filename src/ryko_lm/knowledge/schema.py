"""RYKO Knowledge Core — schema and constants.

A single-file SQLite store (stdlib only, no server) that carries every
heritage passage with *page-level provenance*: each passage points back to
``book -> page -> seq`` and holds the sha256 of its normalized text, so any
generated claim can be re-checked offline.

The schema mirrors the PostgreSQL blueprint of the RYKO Knowledge & Reasoning
Engine research document (``sources / authors / books / passages / claims /
argument_relations``).  ``docs/KNOWLEDGE_CORE.md`` carries the Postgres DDL
(incl. ``pgvector``) for the multi-user scale-out path; SQLite is the local
authoritative store for now.

Design rules:

* **Deterministic ids** — ``passage_id = "p-" + blake2b(book|page|seq)[:10]``;
  re-ingesting the same data reproduces the same ids byte for byte.
* **Rights first** — every source row declares ``license`` and
  ``rights_status`` (``public-domain`` / ``declared-open`` / ``licensed`` /
  ``unknown``); passages inherit at ingest time.
* **Verification status** — passages start ``unverified``; the citation
  validator and later review gates upgrade them.
"""

from __future__ import annotations

import hashlib

SCHEMA_VERSION = 1

# Rights vocabulary (kept intentionally small and closed).
RIGHTS_PUBLIC_DOMAIN = "public-domain"
RIGHTS_DECLARED_OPEN = "declared-open"
RIGHTS_LICENSED = "licensed"
RIGHTS_UNKNOWN = "unknown"

VERIFICATION_UNVERIFIED = "unverified"
VERIFICATION_VERIFIED = "verified"
VERIFICATION_PARTIAL = "partial"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    source_id     TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    homepage      TEXT,
    license       TEXT NOT NULL,
    rights_status TEXT NOT NULL,
    meta          TEXT
);

CREATE TABLE IF NOT EXISTS authors (
    author_id        TEXT PRIMARY KEY,
    name_ar          TEXT NOT NULL,
    death_year_hijri INTEGER,
    meta             TEXT
);

CREATE TABLE IF NOT EXISTS books (
    book_id       TEXT PRIMARY KEY,
    source_id     TEXT NOT NULL REFERENCES sources(source_id),
    title_ar      TEXT NOT NULL,
    author_id     TEXT REFERENCES authors(author_id),
    category      TEXT,
    lang          TEXT NOT NULL DEFAULT 'ar',
    rights_status TEXT NOT NULL DEFAULT 'unknown',
    meta          TEXT
);

CREATE TABLE IF NOT EXISTS passages (
    passage_id          TEXT PRIMARY KEY,
    book_id             TEXT NOT NULL REFERENCES books(book_id),
    page_num            INTEGER,
    seq                 INTEGER NOT NULL,
    text                TEXT NOT NULL,
    text_sha256         TEXT NOT NULL,
    chars               INTEGER NOT NULL,
    discourse           TEXT,
    verification_status TEXT NOT NULL DEFAULT 'unverified',
    meta                TEXT
);
CREATE INDEX IF NOT EXISTS ix_passages_book ON passages(book_id, page_num, seq);
CREATE INDEX IF NOT EXISTS ix_passages_sha  ON passages(text_sha256);

CREATE TABLE IF NOT EXISTS passage_vec (
    passage_id TEXT PRIMARY KEY REFERENCES passages(passage_id),
    vec        BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS claims (
    claim_id     TEXT PRIMARY KEY,
    book_id      TEXT REFERENCES books(book_id),
    passage_id   TEXT REFERENCES passages(passage_id),
    claim_text   TEXT NOT NULL,
    claim_type   TEXT,
    stance       TEXT,
    confidence   REAL,
    meta         TEXT
);

CREATE TABLE IF NOT EXISTS argument_relations (
    relation_id       TEXT PRIMARY KEY,
    claim_id          TEXT NOT NULL REFERENCES claims(claim_id),
    related_claim_id  TEXT NOT NULL REFERENCES claims(claim_id),
    relation          TEXT NOT NULL,
    evidence          TEXT,
    meta              TEXT
);
"""

# Closed vocabulary for ``argument_relations.relation`` (research doc §schema).
RELATIONS = (
    "supports",      # يدعم
    "objects",       # يعترض
    "responds",      # يرد على
    "cites",         # ينقل عن
    "prefers",       # يرجّح
    "summarizes",    # يلخّص
    "paraphrases",   # يعيد الصياغة
    "contradicts",   # يناقض
)


def stable_id(prefix: str, *parts: object) -> str:
    """Deterministic short id: ``prefix-<blake2b(parts)[:10]``."""
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return f"{prefix}-{hashlib.blake2b(payload, digest_size=8).hexdigest()[:10]}"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
