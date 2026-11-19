"""KnowledgeStore — the SQLite-backed authoritative store of the Knowledge Core.

One file, stdlib only.  Carries sources / authors / books / passages /
claims / argument_relations with page-level provenance, plus an in-process
BM25 index and persisted hashing-embedding vectors for hybrid retrieval.

Search contract (deterministic):

* ``mode="bm25"``   — Okapi BM25 (k1=1.5, b=0.75) over the Arabic token
  stream of :mod:`ryko_lm.knowledge.textproc`.
* ``mode="dense"``  — cosine over persisted ``blake2b`` feature-hashing
  vectors (:mod:`ryko_lm.knowledge.embeddings`).
* ``mode="hybrid"`` — reciprocal-rank fusion (k=60) of the two lists.

All modes sort by ``(-score, passage_id)`` so equal scores never shuffle.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import sqlite3
from array import array
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ryko_lm.knowledge import embeddings as emb
from ryko_lm.knowledge import schema as sch
from ryko_lm.knowledge import textproc
from ryko_lm.knowledge.embeddings import DIM, embed, pack, unpack

BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60


def _index_fingerprint(rows: list[dict]) -> str:
    """Stable fingerprint of the passage population for the sidecar cache.

    sha256 over (schema_version, every passage_id in ``rows`` order, every
    text length) — any ingest, delete or text edit invalidates the cache.
    """
    h = hashlib.sha256()
    h.update(f"v{sch.SCHEMA_VERSION}|n{len(rows)}".encode())
    for r in rows:
        h.update(r["passage_id"].encode())
        h.update(str(len(r["text"])).encode())
    return h.hexdigest()


@dataclass(frozen=True)
class Hit:
    passage_id: str
    book_id: str
    title_ar: str
    source_id: str
    page_num: int | None
    seq: int
    text: str
    score: float
    bm25_rank: int | None
    dense_rank: int | None

    def as_dict(self) -> dict:
        return {
            "passage_id": self.passage_id,
            "book_id": self.book_id,
            "title_ar": self.title_ar,
            "source_id": self.source_id,
            "page_num": self.page_num,
            "seq": self.seq,
            "text": self.text,
            "score": round(self.score, 6),
            "bm25_rank": self.bm25_rank,
            "dense_rank": self.dense_rank,
        }


class KnowledgeStore:
    """Open (or create) a knowledge store file; ``:memory:`` for tests.

    ``persist_vectors`` controls the derived ``passage_vec`` table (repo-copy
    policy: derived data never belongs in the committed DB). ``True`` (the
    production default) writes computed embeddings once for warm restarts;
    ``False`` keeps them in-process only — for read-only sessions such as
    tests, one-shot queries against a committed DB, or CI, where a search
    would otherwise dirty the working tree with ~80MB of derived vectors.
    Verification bookkeeping (``set_verification``) is unaffected — it is
    intentional state, not a derived cache.
    """

    def __init__(self, path: str | Path = ":memory:",
                 persist_vectors: bool = True,
                 vector_cache: str | Path | None = None) -> None:
        self.path = Path(path) if path != ":memory:" else None
        self.persist_vectors = persist_vectors
        # Optional derived sidecar for the search index (v1.28.0): a pickle
        # of (tokens, postings, dense matrix) keyed to the exact passage
        # population.  Read-only consumers against the committed repo DB
        # re-derive ~93k passage tokenizations + embeddings on EVERY session
        # (~41s measured); the sidecar turns that into a ~1.5s load while
        # keeping the .sqlite bytes untouched — receipts stay intact.
        self.vector_cache = Path(vector_cache) if vector_cache else None
        # check_same_thread=False: the store may be created in a lifespan
        # thread and used from FastAPI worker threads; access is serialized
        # at the SQL level and the CLI/tests are single-threaded anyway.
        self.conn = sqlite3.connect(
            str(path) if path != ":memory:" else ":memory:",
            check_same_thread=False,
        )
        self.conn.row_factory = sqlite3.Row
        self._ensure_schema()
        self._index: _Index | None = None

    def _ensure_schema(self) -> None:
        """Create/upgrade the schema only when needed.

        Opening an up-to-date store must stay **byte-neutral** (v1.16.0):
        the previous unconditional ``executescript + INSERT OR REPLACE
        meta`` rewrote pages on every open, so even a read-only session
        changed the file's sha256 and broke byte-exact provenance receipts.
        """
        try:
            row = self.conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()
        except sqlite3.OperationalError:
            row = None
        if row is not None:
            try:
                if int(row["value"]) >= int(sch.SCHEMA_VERSION):
                    return
            except (TypeError, ValueError):
                pass
        self.conn.executescript(sch.SCHEMA_SQL)
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(sch.SCHEMA_VERSION),),
        )
        self.conn.commit()

    # ------------------------------------------------------------ writes
    def add_source(self, source_id: str, name: str, license: str,
                   rights_status: str, homepage: str | None = None,
                   meta: dict | None = None) -> None:
        if rights_status not in {
            sch.RIGHTS_PUBLIC_DOMAIN, sch.RIGHTS_DECLARED_OPEN,
            sch.RIGHTS_LICENSED, sch.RIGHTS_UNKNOWN,
        }:
            raise ValueError(f"unknown rights_status: {rights_status!r}")
        self.conn.execute(
            "INSERT OR REPLACE INTO sources VALUES(?,?,?,?,?,?)",
            (source_id, name, homepage, license, rights_status,
             json.dumps(meta, ensure_ascii=False) if meta else None),
        )
        self.conn.commit()
        self._index = None

    def add_author(self, author_id: str, name_ar: str,
                   death_year_hijri: int | None = None,
                   meta: dict | None = None) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO authors VALUES(?,?,?,?)",
            (author_id, name_ar, death_year_hijri,
             json.dumps(meta, ensure_ascii=False) if meta else None),
        )
        self.conn.commit()

    def add_book(self, book_id: str, source_id: str, title_ar: str,
                 author_id: str | None = None, category: str | None = None,
                 rights_status: str = sch.RIGHTS_UNKNOWN,
                 meta: dict | None = None) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO books VALUES(?,?,?,?,?,?,?,?)",
            (book_id, source_id, title_ar, author_id, category, "ar",
             rights_status,
             json.dumps(meta, ensure_ascii=False) if meta else None),
        )
        self.conn.commit()
        self._index = None

    def next_seq(self, book_id: str, page_num: int | None) -> int:
        """Next free ordinal within a (book, page) cell — collision-free ingest."""
        row = self.conn.execute(
            "SELECT COALESCE(MAX(seq), -1) + 1 AS nxt FROM passages "
            "WHERE book_id=? AND page_num IS ?",
            (book_id, page_num),
        ).fetchone()
        return int(row["nxt"])

    def add_passage(self, book_id: str, text: str, page_num: int | None = None,
                    seq: int | None = None, classify: bool = True,
                    meta: dict | None = None) -> tuple[str, bool]:
        """Insert a passage; returns ``(passage_id, created)``.

        ``created=False`` when the same normalized text already exists in the
        same book (in-book dedup).  ``seq`` defaults to the next ordinal of
        the (book, page) cell; a collating caller-provided ``seq`` is bumped
        to the next free ordinal so ids never collide.
        """
        text = text.strip()
        if not text:
            raise ValueError("empty passage text")
        norm = textproc.normalize_for_match(text)
        sha = sch.sha256_text(norm)
        dup = self.conn.execute(
            "SELECT passage_id FROM passages WHERE book_id=? AND text_sha256=?",
            (book_id, sha),
        ).fetchone()
        if dup is not None:
            return dup["passage_id"], False
        discourse = None
        if classify:
            from ryko_lm.knowledge import taxonomy

            label = taxonomy.classify(text)
            discourse = label.code if label else None
        if seq is None:
            seq = self.next_seq(book_id, page_num)
        passage_id = sch.stable_id("p", book_id, page_num if page_num is not None else -1, seq)
        for attempt in range(3):
            try:
                self.conn.execute(
                    "INSERT INTO passages VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (passage_id, book_id, page_num, seq, text, sha, len(text),
                     discourse, sch.VERIFICATION_UNVERIFIED,
                     json.dumps(meta, ensure_ascii=False) if meta else None),
                )
                break
            except sqlite3.IntegrityError:
                # same (book, page, seq) cell already occupied: bump to the
                # next free ordinal — keeps ids deterministic and collision-free
                seq = self.next_seq(book_id, page_num) + attempt
                passage_id = sch.stable_id(
                    "p", book_id, page_num if page_num is not None else -1, seq)
        else:  # pragma: no cover
            raise RuntimeError("passage id collision persisted after retries")
        self.conn.commit()
        self._index = None
        return passage_id, True

    def add_claim(self, claim_text: str, book_id: str | None = None,
                  passage_id: str | None = None, claim_type: str | None = None,
                  stance: str | None = None, confidence: float | None = None,
                  meta: dict | None = None) -> str:
        claim_id = sch.stable_id("c", claim_text[:128], book_id or "", passage_id or "")
        self.conn.execute(
            "INSERT OR REPLACE INTO claims VALUES(?,?,?,?,?,?,?,?)",
            (claim_id, book_id, passage_id, claim_text, claim_type, stance,
             confidence,
             json.dumps(meta, ensure_ascii=False) if meta else None),
        )
        self.conn.commit()
        return claim_id

    def add_relation(self, claim_id: str, related_claim_id: str, relation: str,
                     evidence: str | None = None, meta: dict | None = None) -> str:
        if relation not in sch.RELATIONS:
            raise ValueError(f"unknown relation {relation!r}; allowed: {sch.RELATIONS}")
        relation_id = sch.stable_id("r", claim_id, related_claim_id, relation)
        self.conn.execute(
            "INSERT OR REPLACE INTO argument_relations VALUES(?,?,?,?,?,?)",
            (relation_id, claim_id, related_claim_id, relation, evidence,
             json.dumps(meta, ensure_ascii=False) if meta else None),
        )
        self.conn.commit()
        return relation_id

    # ------------------------------------------------------------- reads
    def stats(self) -> dict:
        def count(table: str) -> int:
            return int(self.conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"])

        by_source = {
            row["source_id"]: int(row["n"])
            for row in self.conn.execute(
                "SELECT b.source_id, COUNT(*) AS n FROM passages p "
                "JOIN books b ON b.book_id = p.book_id GROUP BY b.source_id"
            )
        }
        chars = int(self.conn.execute("SELECT COALESCE(SUM(chars),0) AS n FROM passages").fetchone()["n"])
        return {
            "schema_version": sch.SCHEMA_VERSION,
            "sources": count("sources"),
            "authors": count("authors"),
            "books": count("books"),
            "passages": count("passages"),
            "claims": count("claims"),
            "argument_relations": count("argument_relations"),
            "chars": chars,
            "passages_by_source": by_source,
        }

    def iter_books(self):
        """Deterministic book listing (read-only) — fixed by ``book_id``.

        Read accessor used by, e.g., the benchmark builder's book-attribution
        family (title provenance comes straight from the committed store).
        """
        sql = ("SELECT book_id, source_id, title_ar, author_id, category, "
               "rights_status, meta FROM books ORDER BY book_id")
        for row in self.conn.execute(sql):
            yield dict(row)

    def iter_authors(self):
        """Deterministic author catalog listing (read-only) — ``author_id``.

        Read accessor for the linguistic-attribution distractor pool: the
        closed committed catalog is the only legitimate source of wrong
        originators (no invented attribution may enter a training or eval
        artifact).
        """
        sql = ("SELECT author_id, name_ar, death_year_hijri, meta "
               "FROM authors ORDER BY author_id")
        for row in self.conn.execute(sql):
            yield dict(row)

    def passage(self, passage_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT p.*, b.title_ar, b.source_id FROM passages p "
            "JOIN books b ON b.book_id = p.book_id WHERE p.passage_id=?",
            (passage_id,),
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d.pop("text_sha256", None)
        return d

    def iter_passages(self, book_id: str | None = None):
        sql = ("SELECT p.passage_id, p.book_id, p.text, p.page_num, p.seq FROM passages p "
               "WHERE (? IS NULL OR p.book_id = ?) ORDER BY p.book_id, p.page_num, p.seq")
        for row in self.conn.execute(sql, (book_id, book_id)):
            yield dict(row)

    def iter_claims(self, layer: str | None = None):
        """Deterministic claim listing (read-only) with resolved book + author.

        ``layer`` filters by ``meta.layer`` when provided (e.g. fikr/econom).
        Ordering is fixed by ``claim_id`` — no RNG anywhere.
        """
        sql = ("SELECT c.claim_id, c.book_id, c.passage_id, c.claim_text, "
               "c.claim_type, c.stance, c.meta, b.title_ar, b.author_id, "
               "a.name_ar AS author_name_ar "
               "FROM claims c LEFT JOIN books b ON b.book_id = c.book_id "
               "LEFT JOIN authors a ON a.author_id = b.author_id "
               "WHERE (? IS NULL OR json_extract(c.meta, '$.layer') = ?) "
               "ORDER BY c.claim_id")
        for row in self.conn.execute(sql, (layer, layer)):
            yield dict(row)

    def set_verification(self, passage_id: str, status: str) -> None:
        if status not in {sch.VERIFICATION_UNVERIFIED, sch.VERIFICATION_VERIFIED,
                          sch.VERIFICATION_PARTIAL}:
            raise ValueError(f"unknown verification status: {status!r}")
        self.conn.execute(
            "UPDATE passages SET verification_status=? WHERE passage_id=?",
            (status, passage_id),
        )
        self.conn.commit()

    # ------------------------------------------------------------ search
    def _ensure_index(self) -> "_Index":
        if self._index is None:
            self._index = _Index(self.conn,
                                 persist_vectors=self.persist_vectors,
                                 vector_cache=self.vector_cache)
        return self._index

    def search(self, query: str, k: int = 5, mode: str = "hybrid",
               book_id: str | None = None) -> list[Hit]:
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError(f"unknown mode {mode!r}")
        index = self._ensure_index()
        rows = index.rows
        if book_id is not None:
            # per-book slice from the cached index — no O(n) rescan per call
            rows = [index.rows[i] for i in index.by_book.get(book_id, [])]
            if not rows:
                return []
        results: dict[str, dict] = {}

        def put(pid: str, field: str, rank: int) -> None:
            results.setdefault(pid, {"bm25_rank": None, "dense_rank": None})[field] = rank

        bm25_list: list[tuple[float, dict]] = []
        if mode in {"bm25", "hybrid"}:
            bm25_list = index.bm25_ranked(query, rows)
            # full enumeration in hybrid: a fused top-k hit may carry a
            # bm25_rank beyond k, and those diagnostic fields are part of
            # the byte-equivalence contract with the historical behaviour.
            for rank, (score, row) in enumerate(bm25_list, 1):
                put(row["passage_id"], "bm25_rank", rank)
        dense_list: list[tuple[float, dict]] = []
        if mode in {"dense", "hybrid"}:
            dense_list = index.dense_ranked(query, rows)
            for rank, (score, row) in enumerate(dense_list, 1):
                put(row["passage_id"], "dense_rank", rank)

        if mode == "bm25":
            ordered = bm25_list
            score_of = lambda s, _r: s  # noqa: E731
        elif mode == "dense":
            ordered = dense_list
            score_of = lambda s, _r: s  # noqa: E731
        else:
            fused: dict[str, float] = {}
            for rank, (_s, row) in enumerate(bm25_list, 1):
                fused[row["passage_id"]] = fused.get(row["passage_id"], 0.0) + 1.0 / (RRF_K + rank)
            for rank, (_s, row) in enumerate(dense_list, 1):
                fused[row["passage_id"]] = fused.get(row["passage_id"], 0.0) + 1.0 / (RRF_K + rank)
            # ascending tuple sort on (-score, passage_id) — no lambda key
            fused_items = sorted((-s, pid) for pid, s in fused.items())
            top = fused_items[:k]
            hits: list[Hit] = []
            row_by_id = {r["passage_id"]: r for _s, r in bm25_list + dense_list}
            for neg_score, pid in top:
                row = row_by_id[pid]
                info = results.get(pid, {})
                hits.append(Hit(
                    passage_id=pid, book_id=row["book_id"],
                    title_ar=row["title_ar"], source_id=row["source_id"],
                    page_num=row["page_num"], seq=row["seq"], text=row["text"],
                    score=-neg_score,
                    bm25_rank=info.get("bm25_rank"), dense_rank=info.get("dense_rank"),
                ))
            return hits

        hits: list[Hit] = []
        for score, row in ordered[:k]:
            info = results.get(row["passage_id"], {})
            hits.append(Hit(
                passage_id=row["passage_id"], book_id=row["book_id"],
                title_ar=row["title_ar"], source_id=row["source_id"],
                page_num=row["page_num"], seq=row["seq"], text=row["text"],
                score=score_of(score, row),
                bm25_rank=info.get("bm25_rank"), dense_rank=info.get("dense_rank"),
            ))
        return hits

    def close(self) -> None:
        self.conn.close()


class _Index:
    """In-process BM25 + dense index, rebuilt lazily after writes.

    v1.14.0 performance hardening (same results, same orderings — the
    arithmetic per row is unchanged, only the iteration scheme is):

    * BM25 scores via a token → ``[(row, tf)]`` postings map, so a query
      touches only rows containing at least one query token instead of
      scanning all rows; the per-row term accumulation keeps the query-token
      order, so every score is bit-identical to the row-major original.
    * Dense ranking via a lazy float64 matrix (``rows × DIM``) and one
      matrix–vector product instead of 82K Python-level dot products.
    * Cached row positions and per-book row slices — no per-call O(n) maps.
    """

    def __init__(self, conn: sqlite3.Connection,
                 persist_vectors: bool = True,
                 vector_cache: Path | None = None) -> None:
        self.rows = [
            dict(r) for r in conn.execute(
                "SELECT p.passage_id, p.book_id, p.page_num, p.seq, p.text, "
                "b.title_ar, b.source_id, v.vec FROM passages p "
                "JOIN books b ON b.book_id = p.book_id "
                "LEFT JOIN passage_vec v ON v.passage_id = p.passage_id "
                "ORDER BY p.passage_id"
            )
        ]
        self.pos: dict[str, int] = {
            r["passage_id"]: i for i, r in enumerate(self.rows)
        }
        self.by_book: dict[str, list[int]] = {}
        for i, r in enumerate(self.rows):
            self.by_book.setdefault(r["book_id"], []).append(i)
        self._cache_fp = _index_fingerprint(self.rows)
        if vector_cache is not None and self._try_load_cache(vector_cache):
            return  # sidecar hit: tokens/postings/mat adopted as-is
        self.tokens: list[list[str]] = [textproc.ar_tokens(r["text"]) for r in self.rows]
        self.tf: list[dict[str, int]] = []
        df: dict[str, int] = {}
        total = 0
        for toks in self.tokens:
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            self.tf.append(tf)
            for t in tf:
                df[t] = df.get(t, 0) + 1
            total += len(toks)
        self.n = max(len(self.rows), 1)
        self.avgdl = max(total / self.n, 1.0)
        self.idf = {
            t: math.log(1.0 + (self.n - c + 0.5) / (c + 0.5))
            for t, c in df.items()
        }
        # postings: token -> [(row_index, term_freq)] — built from the same
        # tf maps, so BM25 arithmetic is untouched.
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, tf in enumerate(self.tf):
            for t, f in tf.items():
                self.postings.setdefault(t, []).append((i, f))
        # dense vectors: compute + persist once (row-id keyed by passage_id)
        self.vecs: dict[str, array] = {}
        missing: list[tuple[str, bytes]] = []
        for row, toks in zip(self.rows, self.tokens):
            blob = row["vec"]
            if blob is not None:
                try:
                    self.vecs[row["passage_id"]] = emb.unpack(blob, DIM)
                    continue
                except ValueError:
                    pass
            if toks:
                vec = emb.embed(row["text"])
                self.vecs[row["passage_id"]] = vec
                missing.append((row["passage_id"], emb.pack(vec)))
        if missing and persist_vectors:
            conn.executemany(
                "INSERT OR REPLACE INTO passage_vec VALUES(?,?)", missing
            )
            conn.commit()
        self._mat = None  # lazy float64 matrix (see _ensure_mat)
        if vector_cache is not None:
            self._save_cache(vector_cache)

    # ------------------------------------------------------- sidecar cache
    def _try_load_cache(self, path: Path) -> bool:
        """Adopt a derived sidecar index if it matches the DB fingerprint.

        The payload carries the same structures the fresh path builds —
        ``tokens``, ``postings``, the float32 dense matrix in ``rows`` order
        — so search arithmetic is bit-identical to the from-scratch path.
        Any mismatch (schema, population, corruption) falls back to the
        full recompute and re-saves a fresh sidecar.
        """
        try:
            blob = path.read_bytes()
        except OSError:
            return False
        try:
            payload = pickle.loads(blob)  # noqa: S301 — repo-local derived file
            if payload.get("fingerprint") != self._cache_fp:
                return False
            mat32 = np.asarray(payload["mat32"], dtype=np.float32)
            if mat32.shape != (len(self.rows), DIM):
                return False
            self.tokens: list[list[str]] = payload["tokens"]
            if len(self.tokens) != len(self.rows):
                return False
            self.postings: dict[str, list[tuple[int, int]]] = payload["postings"]
            n = max(len(self.rows), 1)
            total = sum(len(toks) for toks in self.tokens)
            self.avgdl = max(total / n, 1.0)
            self.n = n
            self.idf = {
                t: math.log(1.0 + (n - len(pl) + 0.5) / (len(pl) + 0.5))
                for t, pl in self.postings.items()
            }
            self.tf = None  # not needed: bm25 uses tokens[i] lengths + postings
            self.vecs: dict[str, array] = {}
            self._mat = mat32.astype(np.float64)
            return True
        except Exception:  # corrupted/truncated sidecar — recompute
            return False

    def _save_cache(self, path: Path) -> None:
        """Atomically persist the derived index (tmp + replace)."""
        try:
            payload = {
                "fingerprint": self._cache_fp,
                "tokens": self.tokens,
                "postings": self.postings,
                "mat32": self._ensure_mat().astype(np.float32),
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "wb") as fh:
                pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
        except OSError:
            pass  # cache is best-effort; search stays correct without it

    def _ensure_mat(self):
        """Lazily build the (n × DIM) float64 matrix in ``rows`` order."""
        if self._mat is None:
            mat = np.zeros((len(self.rows), DIM), dtype=np.float64)
            for pid, vec in self.vecs.items():
                mat[self.pos[pid]] = np.frombuffer(bytes(vec), dtype=np.float32)
            self._mat = mat
        return self._mat

    def _row_indices(self, rows: list[dict] | None) -> list[int]:
        """Row indices for a (possibly book-filtered) row list.

        ``None`` means the full index — the identity fast path that skips the
        per-call O(n) rebuild the original paid on every search.
        """
        if rows is self.rows or rows is None:
            return list(range(len(self.rows)))
        pos = self.pos
        return [pos[r["passage_id"]] for r in rows if r["passage_id"] in pos]

    def bm25_ranked(self, query: str, rows: list[dict] | None) -> list[tuple[float, dict]]:
        q = [t for t in textproc.ar_tokens(query)]
        if not q:
            return []
        candidates = self._row_indices(rows)
        wanted = set(candidates)
        # token-major accumulation; per-row term order follows ``q`` order,
        # so each row's score is bit-identical to the row-major original.
        scores: dict[int, float] = {}
        for t in q:
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, f in self.postings.get(t, ()):
                if i not in wanted:
                    continue
                dl = len(self.tokens[i])
                term = idf * (f * (BM25_K1 + 1.0)) / (
                    f + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / self.avgdl)
                )
                scores[i] = scores.get(i, 0.0) + term
        if not scores:
            return []
        # rank without a Python comparator sort: rows are ordered by
        # passage_id, so a stable argsort on -score reproduces the exact
        # original ``sorted(key=(-score, passage_id))`` ordering.
        idx_arr = np.fromiter(scores.keys(), np.int64, len(scores))
        val_arr = np.fromiter(scores.values(), np.float64, len(scores))
        val_arr = val_arr[idx_arr.argsort()]
        idx_arr = idx_arr[idx_arr.argsort()]
        keep = val_arr > 0.0
        idx_arr, val_arr = idx_arr[keep], val_arr[keep]
        order = np.argsort(-val_arr, kind="stable")
        return [(float(val_arr[j]), self.rows[int(idx_arr[j])]) for j in order]

    def dense_ranked(self, query: str, rows: list[dict] | None) -> list[tuple[float, dict]]:
        qv = embed(query)
        q = np.frombuffer(bytes(qv), dtype=np.float32).astype(np.float64)
        full = rows is self.rows or rows is None
        if not full:
            idx = self._row_indices(rows)
            if not idx:
                return []
            arr = np.fromiter(idx, dtype=np.int64, count=len(idx))
            sims = self._ensure_mat()[arr] @ q
            row_seq = [self.rows[i] for i in idx]
        else:
            # full-store fast path: one matrix–vector product, no fancy-index copy
            sims = self._ensure_mat() @ q
            row_seq = self.rows
        keep = np.nonzero(sims > 0.0)[0]
        order = keep[np.argsort(-sims[keep], kind="stable")]
        return [(float(sims[j]), row_seq[j]) for j in order]
