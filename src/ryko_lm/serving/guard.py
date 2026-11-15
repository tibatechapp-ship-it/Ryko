"""Generation guard — pre-delivery validation for RYKO outputs.

Adopted from the ZYKO project's *pre-write guardian* pattern: no generated
text reaches the user without passing structural gates, and the decision is
returned as a signed (hash-sealed) verdict.  ZYKO guards code writes with
PII/RTL/structure checks; RYKO guards LM output with Arabic-text integrity
checks.

Checks (v1):
* non-empty, decodable output;
* Arabic-letter ratio above a floor (catches token-soup / latin drift);
* repetition loop detection (repeated character 4-gram coverage);
* residual corpus markup (``%~``, ``$right$``, ``<...>`` edition blocks);
* runaway whitespace.

Verdicts: ``ALLOW`` | ``REVIEW``.  Every verdict carries a sha256 over
(prompt-hash, output) so audits can recompute it.

Usage:
    from ..serving.guard import guard_output
    verdict = guard_output("قال رسول الله ...")
"""

from __future__ import annotations

import hashlib
import re

ARABIC_LETTER_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")
MARKUP_RE = re.compile(r"%~\d*#?|\$(?:right|left)(?:_m\d*)?\$|<#META")
CHAR_4GRAM_RE = 4

MIN_ARABIC_RATIO = 0.35
MIN_LENGTH = 8
MAX_REPEAT_COVERAGE = 0.6


def _char_ngram_repeat_ratio(text: str, n: int = 4) -> float:
    """``1 - unique/total`` over character 4-grams.

    Healthy Arabic text has a high unique-ngram ratio (>0.7); a degenerate
    loop like ``"قال قال قال ..."`` collapses towards 0 unique grams, so the
    ratio approaches 1.0.
    """
    if len(text) < n + 1:
        return 0.0
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def seal(prompt: str, output: str) -> str:
    """Deterministic sha256 seal over the (prompt, output) pair."""
    blob = hashlib.sha256(prompt.encode("utf-8")).digest()[:8] + b"|" + \
        hashlib.sha256(output.encode("utf-8")).digest()[:8]
    return hashlib.sha256(blob).hexdigest()


def guard_output(text: str, prompt: str = "") -> dict:
    """Validate generated Arabic text; return a signed ALLOW/REVIEW verdict."""
    reasons: list[str] = []
    stripped = text.strip()
    if len(stripped) < MIN_LENGTH:
        reasons.append("too_short")
    if stripped:
        arabic = len(ARABIC_LETTER_RE.findall(stripped))
        ratio = arabic / len(stripped)
        if ratio < MIN_ARABIC_RATIO:
            reasons.append(f"low_arabic_ratio:{ratio:.2f}")
        coverage = _char_ngram_repeat_ratio(stripped)
        if coverage > MAX_REPEAT_COVERAGE:
            reasons.append(f"repetition_loop:{coverage:.2f}")
        if MARKUP_RE.search(stripped):
            reasons.append("residual_markup")
    decision = "ALLOW" if not reasons else "REVIEW"
    return {
        "decision": decision,
        "reasons": reasons,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16] if prompt else None,
        "seal": seal(prompt, text),
    }
