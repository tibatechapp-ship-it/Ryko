"""Deterministic zero-dependency embedder for the Knowledge Core.

Adapted from the ZYKO ``rag.embeddings`` pattern (github.com/tibatechapp-ship-it/zyko):
feature hashing (the *hashing trick*) over ``blake2b`` — each Arabic token is
hashed into a ``(bucket, sign)`` pair, counts accumulate sign-weighted and the
vector is L2-normalized.  Semantically blind but byte-deterministic across
machines and processes: no model download, no API key, no RNG — the same
contract as RYKO receipts.

Vectors are 256-dim float32, packed little-endian into a BLOB so the SQLite
store can persist them and skip recomputation on reopen.
"""

from __future__ import annotations

import hashlib
import math
import struct
from array import array

from ryko_lm.knowledge.textproc import ar_tokens

DIM = 256


def embed(text: str, dim: int = DIM) -> array:
    """Hash text into a normalized ``dim``-dim vector (pure stdlib)."""
    if dim < 8:
        raise ValueError("dim must be >= 8")
    vec = [0.0] * dim
    for tok in ar_tokens(text):
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        bucket = value % dim
        sign = 1.0 if (value >> 63) & 1 else -1.0
        vec[bucket] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return array("f", vec)


def cosine(a: array, b: array) -> float:
    """Cosine similarity for two equal-length float arrays."""
    if len(a) != len(b):
        raise ValueError("vector length mismatch")
    num = 0.0
    for x, y in zip(a, b):
        num += x * y
    return num


def pack(vec: array) -> bytes:
    """float32 vector -> little-endian bytes (host-independent)."""
    return struct.pack(f"<{len(vec)}f", *vec)


def unpack(blob: bytes, dim: int = DIM) -> array:
    """Little-endian bytes -> float32 vector."""
    count = len(blob) // 4
    if count != dim:
        raise ValueError(f"expected {dim} dims, got {count}")
    return array("f", struct.unpack(f"<{count}f", blob))
