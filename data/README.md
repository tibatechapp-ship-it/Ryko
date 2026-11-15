# Ryko corpus

The full corpus is **not committed** (it is re-collectable and large).  This
directory only carries a committed **sample** used by the CI smoke pipeline and
by offline tests:

| file | size | content |
|---|---|---|
| `train.txt` | ~300 KB | paragraphs sampled from the cleaned full corpus |
| `val.txt` | ~80 KB | held-out paragraphs |

## Full corpus provenance (reproducible via `make collect`)

| source | dataset | records | raw volume | register |
|---|---|---|---|---|
| `wikipedia` | `wikimedia/wikipedia` (20231101.ar) via HF datasets-server | 4,908 articles | 36.7 MB | contemporary MSA |
| `tashkeela` | `arbml/tashkeela` via HF datasets-server | 25,700 snippets | 12.5 MB | classical / heritage |

Cleaning (`make clean-corpus`, `ryko_lm.data.clean_corpus`):

1. NFKC unicode folding (presentation forms -> canonical letters),
2. markup / URL / e-mail stripping, wiki citation and file-namespace removal,
3. Arabic orthography: tatweel removal + diacritics removal
   (harakat, dagger alif, Quranic annotation marks),
4. line filters (>= 40 chars, >= 20 Arabic letters, >= 55% Arabic ratio),
5. paragraph-level exact deduplication (6,674 paragraphs dropped),
6. seeded 99/1 train/val split.

Resulting training corpus: **24,098 documents, 22.7M characters, 5.38M tokens**
(see `docs/TRAINING.md` for the token statistics produced by the trained
tokenizer).
