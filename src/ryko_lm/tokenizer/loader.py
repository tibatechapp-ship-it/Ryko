"""High-level wrapper around the trained Ryko tokenizer."""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


class RykoTokenizer:
    """Convenience wrapper: encode/decode with explicit special-token control.

    The underlying ``tokenizers.Tokenizer`` stays framework-agnostic; this
    wrapper only standardizes special-token handling for the LM pipeline.
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------ load
    @classmethod
    def from_file(cls, path: str | Path) -> "RykoTokenizer":
        return cls(Tokenizer.from_file(str(path)))

    @classmethod
    def from_dir(cls, directory: str | Path) -> "RykoTokenizer":
        directory = Path(directory)
        return cls.from_file(directory / "ryko-tokenizer.json")

    # ------------------------------------------------------------- properties
    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size(with_added_tokens=True)

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    # ---------------------------------------------------------------- public
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.tokenizer.encode(text).ids
        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        return ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Encode many lines at once (Rust-side multi-threading)."""
        encs = self.tokenizer.encode_batch(list(texts), add_special_tokens=False)
        return [e.ids for e in encs]

    def decode(self, ids, skip_special: bool = True) -> str:
        if skip_special:
            specials = {PAD_ID, BOS_ID, EOS_ID, UNK_ID}
            ids = [i for i in ids if i not in specials]
        return self.tokenizer.decode(list(ids), skip_special_tokens=skip_special)

    def token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text).ids)

    def meta(self, directory: str | Path) -> dict:
        path = Path(directory) / "tokenizer_meta.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
