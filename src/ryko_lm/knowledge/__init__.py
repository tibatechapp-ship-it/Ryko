"""RYKO Knowledge Core — public surface.

    from ryko_lm.knowledge import KnowledgeStore, verify_quote
    store = KnowledgeStore("data/knowledge/ryko.sqlite")
    hits = store.search("الماء الطهور", k=5)
    verdict = verify_quote(store, "الطهو أن يسأل عن الصحيح")

Everything here is stdlib-only and byte-deterministic: same input, same ids,
same scores — the retrieval side of the receipt contract.
"""

from ryko_lm.knowledge.citation import CitationVerdict, verify_quote
from ryko_lm.knowledge.embeddings import DIM, cosine, embed
from ryko_lm.knowledge.ingest import (
    chunk_text,
    ingest_openiti_jsonl,
    ingest_shamela_jsonl,
)
from ryko_lm.knowledge.schema import RELATIONS, SCHEMA_VERSION, stable_id
from ryko_lm.knowledge.store import Hit, KnowledgeStore
from ryko_lm.knowledge.taxonomy import DISCOURSE_TYPES, classify

__all__ = [
    "CitationVerdict",
    "DIM",
    "DISCOURSE_TYPES",
    "Hit",
    "KnowledgeStore",
    "RELATIONS",
    "SCHEMA_VERSION",
    "chunk_text",
    "classify",
    "cosine",
    "embed",
    "ingest_openiti_jsonl",
    "ingest_shamela_jsonl",
    "stable_id",
    "verify_quote",
]
