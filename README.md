<div align="center">

# Ryko

### Arabic decoder-only language model — full pipeline: corpus → tokenizer → training → serving

**A compact, production-grade Arabic language-model stack built on
[`tokenizers`](https://github.com/huggingface/tokenizers) (Rust) and PyTorch —
Byte-Level BPE, RoPE, RMSNorm, SwiGLU and GQA, trained end-to-end and served
through a streaming API with an Arabic web playground.**

[![CI](https://github.com/tibatechapp-ship-it/Ryko/actions/workflows/ci.yml/badge.svg)](https://github.com/tibatechapp-ship-it/Ryko/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)
![Params](https://img.shields.io/badge/params-11.5M%20%7C%2052M%20%7C%20100M-informational)
![Vocab](https://img.shields.io/badge/vocab-32%2C768%20Byte--BPE-orange)

النسخة العربية: [README_AR.md](README_AR.md)

</div>

---

## What is this repository?

Ryko is a **complete, reproducible pipeline for building an Arabic language
model from scratch** — not a fine-tune of an existing checkpoint:

| stage | module | what it does |
|---|---|---|
| 1. Corpus | `ryko_lm.data.collect_corpus`, `collect_openiti`, `collect_shamela` | contemporary MSA (Arabic Wikipedia) + classical heritage: **Tashkeela, OpenITI 2023.1.8 (mARkdown → plain), Shamela v4 (page-level provenance)** |
| 2. Cleaning | `ryko_lm.data.clean_corpus`, `augment_corpus` | NFKC folding, tatweel/diacritics removal, mARkdown stripping, line filters, streaming global dedup, incremental train/val split |
| 3. Tokenizer | `ryko_lm.tokenizer` | trains a **Byte-Level BPE (32,768)** with an Arabic-aware normalizer — no `[unk]` is ever possible |
| 4. Binarization | `ryko_lm.data.binarize` | corpus → flat `uint16` memmap arrays (nanoGPT-style data path), Rust-side batch encoding |
| 5. Model | `ryko_lm.model.ryko_model` | decoder-only transformer: RoPE, RMSNorm, SwiGLU, GQA, KV-cache decoding |
| 6. Training | `ryko_lm.training.train` | AdamW + cosine schedule + warmup, resumable checkpoints, JSONL metrics |
| 7. Evaluation | `ryko_lm.eval.evaluate` | deterministic sequential-window perplexity + seeded generation samples |
| 8. Serving | `ryko_lm.serving.api` | FastAPI: `/generate`, SSE `/generate/stream`, and an RTL Arabic playground |
| 9. Assurance | `ryko_lm.provenance`, `ryko_lm.serving.guard` | signed sha256 provenance receipts (offline-verifiable) + pre-delivery generation guard (ZYKO pattern) |

The reference model, **Ryko-Tiny (11.5M parameters)**, was trained end-to-end
on CPU inside this repository's CI budget. Two larger presets (`small`, `base`)
ship as ready-to-run GPU configs.

## Results (reference run — Ryko-Tiny v1.2, heritage corpus)

| metric | value |
|---|---|
| Parameters | **11,536,640** (3.15M non-embedding) |
| Architecture | 4 layers · d=256 · 8 heads / 4 KV heads (GQA) · SwiGLU 768 · RoPE θ=10⁴ · ctx 256 |
| Vocabulary | 32,768 Byte-Level BPE, trained in 10.4 s |
| Tokenizer fertility | **1.325 tokens/word** · 4.23 chars/token (Arabic validation text) |
| Training corpus (v3) | **246.1M tokens** (train) + 1.36M (val) — Arabic Wikipedia + Tashkeela + **OpenITI 2023.1.8** (2 parquet shards, 1,782 books) + **Shamela v4** (144 books, 41 categories, page-level provenance); 1.43M duplicate paragraphs dropped (cross-edition dedup) |
| Chinchilla check | 11.5M params × 20 → corpus now **exceeds** the ~230M-token optimal ratio for Ryko-Tiny |
| Training | 5,349 steps · batch 16×256 · **21.9M tokens seen** (resumable; schedule targets 20K steps / 82M tokens) |
| Held-out loss (full-split, same protocol) | before **6.224 (PPL 504.7)** → after **5.078 (PPL 160.4)** — **3.15× better** |
| Inference | ~5 ms/token (CPU, KV cache) · SSE streaming |

**Generation quality (v1.2, heritage domain, temperature 0.85):**

| prompt | sample (abridged) |
|---|---|
| `قال رسول الله صلى الله عليه وسلم` | "...أخبرنا محمد بن هارون ثنا سعيد عن يحيى بن أبي كثير عن عكرمة عن ابن عباس..." — the model now emits **isnad-chain structure** it never saw in v1.1 |
| `فأجاب أهل المذهب بأن` | "...في البيع أو غيره ولا يعتق فيه على بيعها وإنما ي..." — fiqh-discussion register |

Full curves, eval JSONs and raw samples: [docs/results/](docs/results/).

> **Honest engineering note:** at 11.5M parameters and 21.9M training tokens the
> model produces locally coherent classical-register Arabic (isnad patterns,
> fiqh phrasing) but is still **not** semantically reliable — PPL ≈ 160 is a
> 3.15× improvement over v1.1 yet far from production quality. Ryko is a
> **reference pipeline** whose value is the full reproducible stack. The
> resumable trainer + 246M-token corpus mean quality now scales by re-running
> the same command (`docs/ROADMAP.md` phase 3) or switching to `ryko-small`
> (52M). Details: [docs/TRAINING.md](docs/TRAINING.md).

Artifacts committed in-tree: `artifacts/tokenizer/ryko-tokenizer.json`,
`artifacts/training/final.pt` (fp16), training curve and samples in
`docs/results/`.

## Quickstart

```bash
git clone https://github.com/tibatechapp-ship-it/Ryko.git && cd Ryko
pip install -r requirements.txt          # torch (CPU wheels fine) + tokenizers + fastapi

# 1) use the trained artifact directly (no training needed)
python -m ryko_lm.training.evaluate --checkpoint artifacts/training/final.pt \
    --prompts "اللغة العربية" "كان يا ما كان"

# 2) launch the Arabic web playground
make serve                               # http://localhost:8000

# 3) reproduce the full pipeline from scratch
make collect && make clean-corpus && make tokenizer && make binarize && make train
```

Python API:

```python
import torch
from ryko_lm.model.ryko_model import RykoLM
from ryko_lm.tokenizer.loader import RykoTokenizer

model = RykoLM.load("artifacts/training/final.pt")
tok   = RykoTokenizer.from_dir("artifacts/tokenizer")

ids  = tok.encode("اللغة العربية", add_bos=True)
out  = model.generate(torch.tensor([ids]), max_new_tokens=48, temperature=0.8)
print(tok.decode(out[0].tolist()))
```

## Architecture at a glance

```
Byte input ──► Byte-Level BPE (32,768, Arabic normalizer)
        ──► 4 × [ RMSNorm ─► GQA attention (RoPE, SDPA) ─► +residual
                  RMSNorm ─► SwiGLU(768)                    ─► +residual ]
        ──► RMSNorm ─► tied LM head ──► next-token distribution
```

Why these choices for Arabic — Byte-Level BPE over WordPiece/Unigram, 32K
vocabulary, no aggressive letter unification — is analyzed in
[docs/ARABIC_TOKENIZATION.md](docs/ARABIC_TOKENIZATION.md).  The tokenizer
fertility of 1.325 tokens/word (e.g. `فسأكتبها → فس|أ|كتب|ها`,
`فاستخدمناها → ف|است|خد|منا|ها`) shows the BPE learning Arabic clitic
structure instead of fragmenting words.

## Repository layout

```
Ryko/
├── src/ryko_lm/            # the package (data, tokenizer, model, training, serving)
├── configs/models/         # ryko-tiny / ryko-small / ryko-base
├── data/sample/            # committed corpus sample for CI smoke tests
├── artifacts/              # trained tokenizer + final fp16 checkpoint (committed)
├── docs/                   # ARCHITECTURE, ARABIC_TOKENIZATION, TRAINING, results/
├── tests/                  # 30 pytest tests (tokenizer, model, data, serving)
├── archive/                # previous platform iterations (kept for history)
└── .github/workflows/      # CI: unit tests + end-to-end smoke pipeline
```

## Sizing presets

| preset | params | layers | d_model | heads/KV | ffn | context | target hardware |
|---|---|---|---|---|---|---|---|
| `ryko-tiny` | 11.5M | 4 | 256 | 8 / 4 | 768 | 256 | CPU (CI-trained) |
| `ryko-small` | 52.2M | 12 | 512 | 8 / 4 | 1,408 | 512 | 1 consumer GPU |
| `ryko-base` | 100.7M | 12 | 768 | 12 / 4 | 2,048 | 1,024 | 8× GPU (torchrun) |

## Serving

```bash
make serve                                                # http://localhost:8000
curl -s -X POST localhost:8000/generate -H 'Content-Type: application/json' \
     -d '{"prompt": "اللغة العربية", "max_new_tokens": 40, "seed": 7}'
```

Endpoints: `GET /` (RTL Arabic playground) · `GET /health` · `GET /model` ·
`POST /generate` · `POST /generate/stream` (SSE).  All generation knobs are
exposed: temperature, top-k, top-p, repetition penalty, seed.

## Knowledge Core (v1.3.0)

Every heritage passage now carries provenance: a service-free SQLite schema
(`sources / books / passages / claims / argument_relations`), a real seed
(10 books / 11,674 passages with page numbers), deterministic hybrid
retrieval (BM25 + blake2b hashing embeddings + RRF fusion) and a citation
validator returning a signed verdict (`verified / partial / not_found`).
See [docs/KNOWLEDGE_CORE.md](docs/KNOWLEDGE_CORE.md) and
[docs/ROADMAP.md](docs/ROADMAP.md).

```bash
make knowledge-search Q="الإسناد المتصل شرط صحة الحديث"
make knowledge-cite Q="paste the quote to verify"
```

## Roadmap

- [x] v1.1: 6.8× corpus (36.65M tokens), PPL 1,782 → 432 on the same val split
- [x] v1.3: provenance-backed Knowledge Core — hybrid retrieval + citation gate + 10-book seed
- [x] v1.4: evaluation gates — closed ryko-bench contract, deterministic MCQ, abstention, self-checked seed
- [ ] More training tokens (Chinchilla-optimal for 11.5M ≈ 230M tokens)
- [ ] Dialect coverage: add social-media corpora (LABR, MSAC) beside MSA
- [ ] `ryko-small` GPU reference training run with published metrics
- [ ] HuggingFace `transformers`-compatible export (safetensors)
- [ ] Instruction-tuning stage (small Arabic SFT corpus)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).  Security policy:
[SECURITY.md](SECURITY.md).  Community: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

[Apache-2.0](LICENSE) — © 2026 Ryko contributors (tibatechapp-ship-it).
