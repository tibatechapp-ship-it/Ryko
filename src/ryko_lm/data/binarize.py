"""Binarize the cleaned corpus into fixed-width uint16 token arrays.

Layout
------
``train.bin`` / ``val.bin`` are flat ``uint16`` arrays of token ids.  The
training loop slices fixed windows of ``block_size`` tokens out of them with
``numpy.memmap`` — zero-copy, no padding, no attention masks: the classic
nanoGPT-style data path, which keeps CPU training simple and fast.

Special tokens: every paragraph (one per line in ``train.txt``) ends with
``<eos>``; documents (blank-line separated) start with ``<bos>``.

Usage:
    python -m ryko_lm.data.binarize \
        --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
        --in-dir data/processed --out-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from ..tokenizer.loader import BOS_ID, EOS_ID, RykoTokenizer


def _encode_file(tok: RykoTokenizer, path: Path, out_path: Path,
                 batch_lines: int = 2000,
                 state_path: Path | None = None) -> int:
    """Encode ``path`` and stream uint16 tokens to ``out_path`` (memory-safe).

    Uses Rust-side batch encoding; BOS marks the first line of each document
    (blank-line separated), every line ends with EOS.  Returns token count.

    When ``state_path`` is given the run is *checkpointed*: the text offset and
    accumulated token count are saved every ~64 MB, a killed run (CI/CPU time
    budget) resumes exactly from the last checkpoint (append mode), and the
    state file is removed on completion.
    """
    t0 = time.time()
    start_offset, done_tokens, in_document = 0, 0, False
    start_bin_bytes = 0
    if state_path is not None and state_path.exists():
        try:
            st = json.loads(state_path.read_text())
            start_offset, done_tokens = st["offset"], st["tokens"]
            in_document = bool(st.get("in_doc", False))
            # the binary sink may hold up to one checkpoint window of tokens
            # written *after* the state was saved — truncate back to the exact
            # byte position matching the state so appends never duplicate
            start_bin_bytes = int(st.get("bin_bytes", done_tokens * 2))
            print(f"[binarize] resuming {path.name} from offset {start_offset:,} "
                  f"({done_tokens:,} tokens already encoded)", flush=True)
        except Exception:  # noqa: BLE001 - corrupt state restarts cleanly
            start_offset, done_tokens, in_document = 0, 0, False
            start_bin_bytes = 0

    total = done_tokens
    batch: list[str] = []
    first_flags: list[bool] = []
    last_ckpt = start_offset

    def flush_batch() -> None:
        nonlocal total
        if not batch:
            return
        buf: list[int] = []
        for ids, is_first in zip(tok.encode_batch(batch), first_flags):
            if is_first:
                buf.append(BOS_ID)
            buf.extend(ids)
            buf.append(EOS_ID)
        if buf:
            sink.write(np.asarray(buf, dtype=np.uint16).tobytes())
            total += len(buf)
        batch.clear()
        first_flags.clear()

    if start_offset:
        if start_bin_bytes < out_path.stat().st_size:
            with out_path.open("r+b") as fix:
                fix.truncate(start_bin_bytes)  # drop the uncheckpointed tail
        sink = out_path.open("ab")
    else:
        sink = out_path.open("wb")
    try:
        with path.open(encoding="utf-8") as src, sink:
            if start_offset:
                src.seek(start_offset)
            else:
                pass
            while True:
                raw = src.readline()
                if not raw:  # EOF
                    break
                if not raw.strip():  # document separator
                    in_document = False
                    continue
                first_flags.append(not in_document)
                batch.append(raw.rstrip("\n"))
                in_document = True
                if len(batch) >= batch_lines:
                    flush_batch()
                    if state_path is not None:
                        pos = src.tell()
                        if pos - last_ckpt >= 64_000_000:
                            state_path.write_text(json.dumps(
                                {"offset": pos, "tokens": total,
                                 "in_doc": in_document,
                                 "bin_bytes": sink.tell()}))
                            last_ckpt = pos
            flush_batch()
            if state_path is not None:
                state_path.write_text(json.dumps(
                    {"offset": path.stat().st_size, "tokens": total,
                     "in_doc": in_document, "bin_bytes": sink.tell()}))
    finally:
        if state_path is not None and state_path.exists():
            st = json.loads(state_path.read_text())
            if st.get("offset") == path.stat().st_size:
                state_path.unlink()
    print(f"[binarize] {path.name}: {total:,} tokens in {time.time() - t0:.1f}s", flush=True)
    return total


def binarize(tokenizer_path: Path, in_dir: Path, out_dir: Path) -> dict:
    tok = RykoTokenizer.from_file(tokenizer_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats: dict = {"vocab_size": tok.vocab_size}
    for split in ("train", "val"):
        src = in_dir / f"{split}.txt"
        if not src.exists():
            continue
        out = out_dir / f"{split}.bin"
        state_path = out_dir / f".{split}.binarize_state.json"
        if out.exists() and not state_path.exists() and out.stat().st_mtime > src.stat().st_mtime:
            # fully encoded by an earlier process (no pending state) — skip
            n_tokens = out.stat().st_size // 2
            print(f"[binarize] {split}.bin already complete ({n_tokens:,} tokens) — skip", flush=True)
            stats[split] = {"tokens": int(n_tokens), "file": str(out)}
            continue
        n_tokens = _encode_file(tok, src, out, state_path=state_path)
        stats[split] = {"tokens": int(n_tokens), "file": str(out)}

    (out_dir / "binarize_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] {json.dumps({k: v for k, v in stats.items() if k != 'vocab_size'})}", flush=True)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Binarize cleaned corpus to uint16 arrays.")
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer/ryko-tokenizer.json"))
    parser.add_argument("--in-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args(argv)
    binarize(args.tokenizer, args.in_dir, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
