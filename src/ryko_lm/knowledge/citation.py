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
from dataclasses import dataclass

from ryko_lm.knowledge import textproc
from ryko_lm.knowledge.schema import VERIFICATION_PARTIAL, VERIFICATION_VERIFIED
from ryko_lm.knowledge.store import KnowledgeStore

PROTOCOL = "ryko-cite/1"


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
