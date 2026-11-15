"""Train the Ryko Arabic tokenizer (Byte-Level BPE, 32,768 vocabulary).

Why Byte-Level BPE for Arabic?
    * operates on raw bytes -> no ``[unk]`` is ever possible, whatever the
      input (diacritics, Latin brand names, emojis, math symbols);
    * degrades gracefully to byte fallback for rare morphology instead of
      fragmenting rare Arabic words the way a character-level WordPiece would;
    * byte fallback also makes mixed Arabic/Latin/numeric web text safe.

The normalizer (``ryko_lm.tokenizer.normalizers``) mirrors the corpus cleaning
cascade, so diacritized input encodes to the same tokens as clean input.

Usage:
    python -m ryko_lm.tokenizer.train_tokenizer \
        --corpus data/processed/train.txt --out artifacts/tokenizer
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .normalizers import build_normalizer

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


def build_tokenizer(vocab_size: int, min_frequency: int, unify_alef: bool = False) -> Tokenizer:
    """Return an untrained BPE tokenizer with the Ryko Arabic front-end."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = build_normalizer(unify_alef=unify_alef)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def train(
    corpus_files: list[Path],
    out_dir: Path,
    vocab_size: int = 32768,
    min_frequency: int = 2,
    unify_alef: bool = False,
) -> Tokenizer:
    """Train the BPE merges and save the tokenizer JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer(vocab_size, min_frequency, unify_alef)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )

    t0 = time.time()
    files = [str(p) for p in corpus_files]
    tokenizer.train(files, trainer)
    elapsed = time.time() - t0

    final_size = tokenizer.get_vocab_size(with_added_tokens=True)
    print(f"[tokenizer] trained {final_size} tokens in {elapsed:.1f}s", flush=True)

    out_json = out_dir / "ryko-tokenizer.json"
    tokenizer.save(str(out_json))
    meta = {
        "algo": "ByteLevelBPE",
        "vocab_size": final_size,
        "min_frequency": min_frequency,
        "special_tokens": SPECIAL_TOKENS,
        "normalizer": "NFKC + strip(tatweel|harakat|dagger-alif|quran-marks)"
        + (" + unify-alef" if unify_alef else ""),
        "corpus_files": [p.name for p in corpus_files],
        "train_seconds": round(elapsed, 1),
    }
    (out_dir / "tokenizer_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[tokenizer] saved -> {out_json}", flush=True)
    return tokenizer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the Ryko Arabic tokenizer.")
    parser.add_argument("--corpus", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/tokenizer"))
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--unify-alef", action="store_true")
    args = parser.parse_args(argv)

    for path in args.corpus:
        if not path.exists():
            print(f"[error] corpus file not found: {path}", file=sys.stderr)
            return 1

    train(
        corpus_files=args.corpus,
        out_dir=args.out,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        unify_alef=args.unify_alef,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
