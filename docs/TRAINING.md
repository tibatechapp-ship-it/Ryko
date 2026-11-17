# Training Ryko — reference run and reproduction guide

## Reference run: Ryko-Tiny v1.1 on CPU

Everything below was produced inside this repository with the committed
pipeline, on 2 CPU cores, across resumable chunks (`--time-budget` + an
auto-restart wrapper that survives host OOM kills):

| item | value |
|---|---|
| config | `configs/models/ryko-tiny.yaml` |
| optimizer | AdamW (β 0.9/0.95), wd 0.1, clip 1.0 |
| schedule | warmup 100 → cosine 3.0e-4 → 3.0e-5 over 4,000 steps |
| batch | 16 sequences × 256 tokens = 4,096 tokens/step (`--batch-size 16`) |
| corpus | **36.65M-token** train split — Arabic Wikipedia deep slices + Tashkeela |
| total tokens seen | **16,384,000** (0.45 epoch) |
| wall time | ~55 minutes on 2 CPU cores (~5,200 tokens/s steady-state) |
| final | train 6.11 · val 6.07 · **PPL 432** |
| vs v1.0, same val split | val 7.49 → **6.07** nats · PPL 1,782 → **432 (−76%)** |

### Loss curve

![loss curve](results/loss_curve.png)

| step | train | val | lr | tokens seen |
|---:|---:|---:|---:|---:|
| 400 | 7.331 | 7.492 | 2.96e-4 | 1,638,400 |
| 800 | 7.028 | 6.957 | 2.79e-4 | 3,276,800 |
| 1200 | 6.782 | 6.664 | 2.50e-4 | 4,915,200 |
| 1600 | 6.533 | 6.400 | 2.13e-4 | 6,553,600 |
| 2000 | 6.650 | 6.430 | 1.71e-4 | 8,192,000 |
| 2400 | 6.256 | 6.266 | 1.28e-4 | 9,830,400 |
| 2800 | 6.195 | 6.222 | 8.84e-5 | 11,468,800 |
| 3200 | 6.115 | 6.125 | 5.71e-5 | 13,107,200 |
| 3600 | 6.070 | 6.136 | 3.70e-5 | 14,745,600 |
| 4000 | 6.113 | 6.057 | 3.00e-5 | 16,384,000 |

### Interpretation (read this)

* The v1.1 curve is textbook-healthy: val tracks train within ~0.1 nat all the
  way down, no divergence at 0.45 epoch of a 36.65M-token corpus.
* PPL ≈ 432 is still **high, and honest**: an 11.5M model seeing 16.4M tokens
  is far below compute-optimal (Chinchilla suggests ≈ 230M tokens for this
  size). But the *style* change is dramatic — the model now emits coherent
  Wikipedia-style article openings and city-stub paragraphs instead of
  word salad.
* The pipeline is the product; scale is a config change.

### Samples (temperature 0.8)

Full JSON: [results/samples.json](results/samples.json).  Before/after on
identical prompts (v1.0 PPL 1,502 → v1.1 PPL 432):

```
prompt: كان يا ما كان
v1.0:   هو هو الأول أن يقال له ذلك وإن كان لهم أنه لا يجوز في رواية ابن عمر ...
v1.1:   في منطقة سكنية تقع على بعد 40 كم شمال غرب مدينة غولد في ولاية كاليفورنيا،
        التابعة لمحافظة ظفار في سلطنة عمان. يقدر عدد سكانها ب 203 نسمة حسب إحصاء
        عام 2010 التابع للمركز الوطني للإحصاء والمعلومات الحكومي. رمز ...

prompt: اللغة العربية
v1.0:   ، هي مدينة في 15 يوليو 2010. ومن خلال العام عام 2007، كانت هي شركة هيميس ...
v1.1:   ، هي إحدى الدوائر الانتخابية في المملكة المتحدة الممثلة في مجلس العموم
        في برلمان المملكة المتحدة.
```

(The model learned the corpus's templates, not world facts — judge fluency
and structure, not truth.)

## Lessons from training on a flaky 2-core CPU box

The v1.0 → v1.1 gap came from three production-style fixes worth reading:

1. **Resume must advance the data stream.**  The original trainer re-seeded the
   window sampler with a fixed seed on every start, so each resumed session
   replayed *the same windows*.  After dozens of kill-and-resume cycles the
   model had memorised that prefix: logged train loss fell to 3.3 while loss on
   any fresh window sat at 8.4 — invisible overfitting.  The trainer now seeds
   `rng(seed + step)`, making every session sample a fresh stream.
2. **Checkpoint writes must be atomic.**  A kill during `torch.save` truncated
   `last.pt` to 0 bytes and lost the optimizer state.  All saves now go through
   a temp file + `os.replace`, so a checkpoint is either the old one or the new
   one — never a partial file.
3. **Budget stops must checkpoint too.**  `--time-budget` used to stop without
   saving unless an eval step coincided; `save_last()` now runs on every
   graceful stop, and `tokens_seen` is persisted in the checkpoint so changing
   `--batch-size` mid-run keeps the accounting exact.

## Reproduce

```bash
# full reference pipeline (≈ 2 h CPU total; the tokenizer step takes 10 s)
make collect && make clean-corpus && make tokenizer && make binarize
make train                 # resumable: rerun until it prints [done]

# evaluate + publish results
make eval && make samples && make plot
```

Resumable chunks (how the reference run was produced) — each run checkpoints
atomically on graceful stop, and the sampling stream advances with the step:

```bash
python -m ryko_lm.training.train --config configs/models/ryko-tiny.yaml \
    --batch-size 16 --time-budget 500   # exits gracefully, rerun to continue
```

## Scaling to GPU configs

| preset | command | notes |
|---|---|---|
| `ryko-small` | `python -m ryko_lm.training.train --config configs/models/ryko-small.yaml` | 1 consumer GPU (24 GB), ~50k steps, bf16 autocast recommended |
| `ryko-base` | `torchrun --nproc_per_node=8 -m ryko_lm.training.train --config configs/models/ryko-base.yaml` | DDP across 8 GPUs; wrap the optimizer with ZeRO/FSDP for comfort |

Data scaling guidance: for a 52M model target ≈ 1B tokens (≈ 200× this corpus
— use the Wikimedia Arabic dump instead of the API sampler, and add
dialect/social corpora: LABR, MSAC, OSIAN).  The cleaning and tokenizer stages
are unchanged; raise `vocab_size` only if you add heavy dialect coverage.

## Artifacts inventory

| path | committed | content |
|---|---|---|
| `artifacts/tokenizer/ryko-tokenizer.json` | yes | trained BPE tokenizer (32,768) |
| `artifacts/tokenizer/tokenizer_meta.json` | yes | training metadata |
| `artifacts/training/final.pt` | yes | final fp16 checkpoint (config + weights) — **v1.1 model** (archive; NOT the serving point since v1.17.0) |
| `artifacts/training_v5c/final.pt` | yes | **serving point since v1.17.0** — v5c shard on corpus v5.2, held-out PPL 221.38, receipted (`final_receipt.json`) |
| `artifacts/training/log.jsonl` | yes | eval metrics every 200 steps (v1.1 run) |
| `artifacts/training/summary.json` | yes | final summary (steps, PPL, params) |
| `artifacts/training/{last,best}.pt` | no | resumable/val-best checkpoints |
| `docs/results/` | yes | loss curve, samples, tokenizer stats |
