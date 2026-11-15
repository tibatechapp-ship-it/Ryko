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

import json
import math
import sqlite3
from array import array
from dataclasses import dataclass
from pathlib import Path

from ryko_lm.knowledge import embeddings as emb
from ryko_lm.knowledge import schema as sch
from ryko_lm.knowledge import textproc
from ryko_lm.knowledge.embeddings import DIM, cosine, embed, pack, unpack

BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60


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
    """Open (or create) a knowledge store file; ``:memory:`` for tests."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = Path(path) if path != ":memory:" else None
        self.conn = sqlite3.connect(str(path) if path != ":memory:" else ":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(sch.SCHEMA_SQL)
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(sch.SCHEMA_VERSION),),
        )
        self.conn.commit()
        self._index: _Index | None = None

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
            self._index = _Index(self.conn)
        return self._index

    def search(self, query: str, k: int = 5, mode: str = "hybrid",
               book_id: str | None = None) -> list[Hit]:
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError(f"unknown mode {mode!r}")
        index = self._ensure_index()
        rows = index.rows
        if book_id is not None:
            rows = [r for r in rows if r["book_id"] == book_id]
            if not rows:
                return []
        results: dict[str, dict] = {}

        def put(pid: str, field: str, rank: int) -> None:
            results.setdefault(pid, {"bm25_rank": None, "dense_rank": None})[field] = rank

        bm25_list: list[tuple[float, dict]] = []
        if mode in {"bm25", "hybrid"}:
            bm25_list = index.bm25_ranked(query, rows)
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
            row_by_id = {r["passage_id"]: r for _s, r in bm25_list + dense_list}
            ordered = sorted(
                ((s, row_by_id[pid]) for pid, s in fused.items()),
                key=lambda t: (-t[0], t[1]["passage_id"]),
            )
            score_of = lambda _s, r: fused[r["passage_id"]]  # noqa: E731

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
    """In-process BM25 + dense index, rebuilt lazily after writes."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.rows = [
            dict(r) for r in conn.execute(
                "SELECT p.passage_id, p.book_id, p.page_num, p.seq, p.text, "
                "b.title_ar, b.source_id, v.vec FROM passages p "
                "JOIN books b ON b.book_id = p.book_id "
                "LEFT JOIN passage_vec v ON v.passage_id = p.passage_id "
                "ORDER BY p.passage_id"
            )
        ]
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
        if missing:
            conn.executemany(
                "INSERT OR REPLACE INTO passage_vec VALUES(?,?)", missing
            )
            conn.commit()

    def bm25_ranked(self, query: str, rows: list[dict]) -> list[tuple[float, dict]]:
        q = [t for t in textproc.ar_tokens(query)]
        if not q:
            return []
        pos = {r["passage_id"]: i for i, r in enumerate(self.rows)}
        scores: dict[str, float] = {}
        for row in rows:
            i = pos[row["passage_id"]]
            tf = self.tf[i]
            dl = len(self.tokens[i])
            if dl == 0:
                continue
            s = 0.0
            for t in q:
                f = tf.get(t)
                if not f:
                    continue
                s += self.idf.get(t, 0.0) * (f * (BM25_K1 + 1.0)) / (
                    f + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / self.avgdl)
                )
            if s > 0:
                scores[row["passage_id"]] = s
        ordered = sorted(
            ((s, r) for r in rows if (s := scores.get(r["passage_id"]))),
            key=lambda t: (-t[0], t[1]["passage_id"]),
        )
        return ordered

    def dense_ranked(self, query: str, rows: list[dict]) -> list[tuple[float, dict]]:
        qv = embed(query)
        out: list[tuple[float, dict]] = []
        for row in rows:
            vec = self.vecs.get(row["passage_id"])
            if vec is None:
                continue
            out.append((cosine(qv, vec), row))
        out = [t for t in out if t[0] > 0.0]
        out.sort(key=lambda t: (-t[0], t[1]["passage_id"]))
        return out
