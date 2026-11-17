"""Citation validator — can a generated claim be anchored to a real passage?

Given a quote (e.g. produced by the model or copied by a reader), the
validator returns a signed-off verdict:

* ``verified``  — the normalized quote appears **verbatim** inside a stored
  passage (normalization = search normalization of ``textproc``).
* ``partial``   — enough of the quote's tokens are found in the best
  candidate (token containment >= ``min_partial``), but not verbatim.
* ``not_found`` — no candidate clears the bar.

Every verdict carries ``quote_sha256`` (of the normalized quote), the
matched ``passage_id`` / ``book_id`` / ``page_num`` and a deterministic
``verified_at`` protocol version, so the verdict itself is auditable and
can be embedded into a provenance receipt.

This is the honesty gate of RYKO: a model output without a ``verified``
citation is presented as *unattributed*, never as a fatwa.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass

from ryko_lm.knowledge import textproc
from ryko_lm.knowledge.schema import VERIFICATION_PARTIAL, VERIFICATION_VERIFIED
from ryko_lm.knowledge.store import KnowledgeStore

PROTOCOL = "ryko-cite/1"

# Sentence splitter: Arabic question/exclamation marks, full stop, newlines
# and the ellipsis — document order preserved (deterministic, no sorting).
_SENTENCE_SPLIT = re.compile(r"[.؟!\n…]+")


@dataclass(frozen=True)
class CitationVerdict:
    status: str            # verified | partial | not_found
    quote_sha256: str
    passage_id: str | None
    book_id: str | None
    title_ar: str | None
    page_num: int | None
    score: float           # 1.0 for verbatim, else token containment
    protocol: str

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "quote_sha256": self.quote_sha256,
            "passage_id": self.passage_id,
            "book_id": self.book_id,
            "title_ar": self.title_ar,
            "page_num": self.page_num,
            "score": round(self.score, 4),
            "protocol": self.protocol,
        }

    def as_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, indent=2)


def verify_quote(store: KnowledgeStore, quote: str, book_id: str | None = None,
                 min_partial: float = 0.82, candidates: int = 25) -> CitationVerdict:
    norm_quote = textproc.normalize_for_match(quote)
    if not norm_quote:
        raise ValueError("empty quote")
    quote_sha = hashlib.sha256(norm_quote.encode("utf-8")).hexdigest()
    q_tokens = set(textproc.ar_tokens(quote))

    best = None  # (score, row)
    if book_id is not None:
        pool = store.search(quote, k=candidates, mode="hybrid", book_id=book_id)
    else:
        pool = store.search(quote, k=candidates, mode="hybrid")

    for hit in pool:
        norm_pass = textproc.normalize_for_match(hit.text)
        if norm_quote and norm_quote in norm_pass:
            best = (1.0, hit)
            break
        if q_tokens:
            p_tokens = set(textproc.ar_tokens(hit.text))
            containment = len(q_tokens & p_tokens) / len(q_tokens)
            if best is None or containment > best[0]:
                best = (containment, hit)

    if best is None:
        return CitationVerdict("not_found", quote_sha, None, None, None, None, 0.0, PROTOCOL)

    score, hit = best
    if score >= 1.0:
        status = VERIFICATION_VERIFIED
    elif score >= min_partial:
        status = VERIFICATION_PARTIAL
    else:
        return CitationVerdict("not_found", quote_sha, None, None, None, None, score, PROTOCOL)
    if status == VERIFICATION_VERIFIED:
        store.set_verification(hit.passage_id, VERIFICATION_VERIFIED)
    return CitationVerdict(status, quote_sha, hit.passage_id, hit.book_id,
                           hit.title_ar, hit.page_num, score, PROTOCOL)


# ---------------------------------------------------- answer audit (v1.15.0)
#
# The generation loop of /knowledge/ask already retrieved its candidate
# passages before generating; auditing the *answer* against that same fixed
# pool costs zero extra retrieval and is fully deterministic. This closes the
# loop promised by COMPETITIVE.md weakness #7: the answer is delivered only
# with an auditable anchor, or the endpoint abstains.

SENT_AUDIT_MIN_CHARS = 10       # same floor as CiteRequest.quote
ANSWER_AUDIT_MAX_SENTENCES = 4  # audit the first N qualifying sentences

AUDIT_ATTRIBUTED = "attributed"      # >= 1 verbatim sentence
AUDIT_PARTIAL = "partial"            # no verbatim, >= 1 >= min_partial
AUDIT_UNATTRIBUTED = "unattributed"  # nothing anchors


@dataclass(frozen=True)
class SentenceVerdict:
    """ryko-cite/1 verdict for one sentence of a generated answer."""

    sentence: str
    status: str            # verified | partial | not_found
    score: float
    passage_id: str | None
    book_id: str | None
    title_ar: str | None
    page_num: int | None

    def as_dict(self) -> dict:
        return {
            "sentence": self.sentence,
            "status": self.status,
            "score": round(self.score, 4),
            "passage_id": self.passage_id,
            "book_id": self.book_id,
            "title_ar": self.title_ar,
            "page_num": self.page_num,
        }


@dataclass(frozen=True)
class AnswerAudit:
    """Aggregate audit of a generated answer against its retrieval pool."""

    protocol: str
    sentences_audited: int
    verdicts: tuple[SentenceVerdict, ...]
    attributed_ratio: float     # verified sentences / audited sentences
    status: str                 # attributed | partial | unattributed
    best: SentenceVerdict | None
    latency_ms: float

    def as_dict(self) -> dict:
        return {
            "protocol": self.protocol,
            "sentences_audited": self.sentences_audited,
            "verdicts": [v.as_dict() for v in self.verdicts],
            "attributed_ratio": round(self.attributed_ratio, 4),
            "status": self.status,
            "best": self.best.as_dict() if self.best else None,
            "latency_ms": round(self.latency_ms, 1),
        }


def _split_sentences(answer: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(answer)]
    return [p for p in parts if len(p) >= SENT_AUDIT_MIN_CHARS]


def _pfield(passage, name: str):
    """Duck-typed field access: str | dict | SearchHit-like object."""
    if isinstance(passage, str):
        return passage if name == "text" else None
    if isinstance(passage, dict):
        return passage.get(name)
    return getattr(passage, name, None)


def audit_answer(answer: str, passages: list, min_partial: float = 0.82,
                 max_sentences: int = ANSWER_AUDIT_MAX_SENTENCES) -> AnswerAudit:
    """Audit a generated answer against a *fixed* candidate pool.

    ``passages`` are the hits already retrieved for this answer (anything
    exposing ``.text/.passage_id/.book_id/.title_ar/.page_num``, or plain
    strings). Each qualifying sentence (>= ``SENT_AUDIT_MIN_CHARS`` chars,
    first ``max_sentences`` in document order) is scored against every
    passage: verbatim normalized containment = 1.0, else token containment.
    No store access, no RNG — deterministic by construction.
    """
    t0 = time.perf_counter()
    passages = list(passages)  # accept generators (e.g. iter_passages) once
    sentences = _split_sentences(answer)[:max_sentences]
    verdicts: list[SentenceVerdict] = []
    best: SentenceVerdict | None = None
    verified = 0
    for sentence in sentences:
        s_tokens = set(textproc.ar_tokens(sentence))
        s_score, s_hit = 0.0, None
        for passage in passages:
            text = _pfield(passage, "text") or ""
            ids = (_pfield(passage, "passage_id"), _pfield(passage, "book_id"),
                   _pfield(passage, "title_ar"), _pfield(passage, "page_num"))
            if s_tokens:
                p_tokens = set(textproc.ar_tokens(text))
                containment = len(s_tokens & p_tokens) / len(s_tokens)
            else:
                containment = 0.0
            norm_sentence = textproc.normalize_for_match(sentence)
            if norm_sentence and norm_sentence in textproc.normalize_for_match(text):
                containment = 1.0
            if containment > s_score or s_hit is None:
                s_score, s_hit = containment, ids
        pid, bid, title, page = s_hit
        if s_score >= 1.0:
            status = VERIFICATION_VERIFIED
            verified += 1
        elif s_score >= min_partial:
            status = VERIFICATION_PARTIAL
        else:
            status = "not_found"
        verdict = SentenceVerdict(sentence, status, s_score, pid, bid, title, page)
        verdicts.append(verdict)
        if best is None or s_score > best.score:
            best = verdict

    if verified:
        status = AUDIT_ATTRIBUTED
    elif any(v.status == VERIFICATION_PARTIAL for v in verdicts):
        status = AUDIT_PARTIAL
    else:
        status = AUDIT_UNATTRIBUTED
    ratio = verified / len(verdicts) if verdicts else 0.0
    latency = (time.perf_counter() - t0) * 1000
    return AnswerAudit(PROTOCOL, len(verdicts), tuple(verdicts), ratio,
                       status, best, latency)
