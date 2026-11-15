# Arabic Tokenization — engineering decisions in Ryko

This document explains **why** the Ryko tokenizer is built the way it is, and
how Arabic morphology drove every choice.  It complements the code in
`src/ryko_lm/tokenizer/`.

## 1. Algorithm choice: Byte-Level BPE

| algorithm | pros | cons for Arabic | verdict |
|---|---|---|---|
| **Byte-Level BPE** | byte fallback → `[unk]` impossible; single pipeline for Arabic/Latin/emoji/math; merges learn frequent morphemes | can over-fragment very rare words (bytes) | **chosen** |
| WordPiece | good with fixed vocab | needs `[unk]` fallback or huge char inventory; breaks on diacritics unless normalized | rejected |
| Unigram (SentencePiece) | probabilistic, often better compression | weaker ecosystem inside `tokenizers` training loop; slower sampling during training | future work |

Byte-Level BPE is what Jais, AceGPT, LLaMA and GPT-family models use.  Ryko
trains it with `tokenizers` (`BpeTrainer`) in ~10 s on 22.7M characters.

Measured quality on Ryko's held-out Arabic text:

* **fertility = 1.325 tokens/word** (multilingual tokenizers often pay 2–4 on Arabic)
* **4.23 characters/token** average compression
* zero `unk`, zero byte-fragmentation on 20K sampled words

## 2. How BPE absorbs Arabic clitics (الإلصاق)

Arabic attaches conjunctions, prepositions, articles, and pronouns directly to
the stem (`وبالدفتر`، `فسأكتبها`، `كتابهما`).  A word-level vocabulary would
need a separate entry for every inflection; a byte-level vocabulary explodes;
BPE instead learns the **clitic affixes** as reusable pieces:

| word | Ryko tokens |
|---|---|
| `وبالدفتر` | `وب` + `الد` + `ف` + `تر` |
| `فسأكتبها` | `فس` + `أ` + `كتب` + `ها` |
| `والكتاب` | `وال` + `كتاب` |
| `بالمدرسة` | `بال` + `مدرسة` |
| `فاستخدمناها` | `ف` + `است` + `خد` + `منا` + `ها` |
| `كتابهما` | `كتاب` + `هما` |

Prefixes (`و`، `ف`، `ب`، `ل`، `ال`) and suffixes (`ها`، `هما`، `نا`، `كم`)
become standalone tokens while roots (`كتب`، `استخدام`) stay intact — exactly
the sub-word behavior that keeps a 32K vocabulary sufficient for Arabic.

## 3. Diacritics (التشكيل) and orthographic normalization

The normalizer cascade (shared by corpus cleaning *and* the tokenizer, so raw
and cleaned text land in the same token space):

1. **NFKC** — folds Arabic presentation forms (`ﻻ → لا`) and legacy glyphs.
2. **Delete** tatweel `ـ` (U+0640), harakat (U+064B–U+065F), dagger alif
   (U+0670), Quranic annotation marks (U+06D6–U+06ED).
3. **Optional (off by default):** unify alef forms (`أ إ آ → ا`), alif
   maqsura (`ى → ي`), taa marbuta (`ة → ه`).

Why not normalize alef/hamza aggressively?  Those letters **carry meaning**
(`أكل` "he ate" vs `اكل`-forms differ; `حالة` vs `حاله`), and modern Arabic
modeling practice (Jais, AceGPT) keeps them.  Removing diacritics is standard
for non-Quranic corpora and shrinks spurious diversity; it stays configurable
(`--keep-diacritics`) for future diacritization experiments — and because the
tokenizer deletes harakat itself, a diacritized user prompt maps to the same
tokens as clean text.

## 4. Vocabulary size: 32,768

* Too small (<16K) → over-tokenization: long words shatter into 4–6 byte
  pieces, sequences inflate, inference slows down.
* Too large (>65K) → the embedding matrix dominates parameters at Ryko's
  scale (32K × 256 = 8.4M of the 11.5M total already) and rare tokens
  under-train.
* 32K matches the common range used by Arabic-first models (Jais-family
  bases, AceGPT) and our measured fertility confirms it is sufficient.

## 5. Pre-tokenization details

`ByteLevel(add_prefix_space=False, use_regex=True)` — the GPT-2 regex groups
Arabic letters into word-like chunks (it treats Arabic as letters, not
punctuation), keeps punctuation separate, and handles mixed-direction text
(Arabic + Latin brand names + digits) deterministically.

## 6. Special tokens

| id | token | role |
|---|---|---|
| 0 | `<pad>` | reserved (data path is padding-free) |
| 1 | `<bos>` | document start (added by the binarizer and the server) |
| 2 | `<eos>` | paragraph/document end + generation terminator |
| 3 | `<unk>` | reserved for API compatibility; never produced |

## 7. Reproducing

```bash
make tokenizer            # trains artifacts/tokenizer/ryko-tokenizer.json
python - <<'PY'
from tokenizers import Tokenizer
tok = Tokenizer.from_file("artifacts/tokenizer/ryko-tokenizer.json")
for w in ["وبالدفتر", "فسأكتبها", "فاستخدمناها"]:
    print(w, "->", [tok.decode([i]) for i in tok.encode(w).ids])
PY
```
