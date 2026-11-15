# Ryko — Arabic Language Model
.PHONY: help collect clean-corpus tokenizer binarize train train-smoke eval samples \
            plot serve test pipeline clean

help:
            @echo "Ryko — Arabic decoder-only language model"
            @echo "  make collect        Collect raw Arabic corpus (Wikipedia + Tashkeela)"
            @echo "  make clean-corpus   Clean/normalize/deduplicate the corpus"
            @echo "  make tokenizer      Train the Byte-Level BPE tokenizer (32,768)"
            @echo "  make binarize       Encode corpus to uint16 token arrays"
            @echo "  make train          Train Ryko-Tiny (resumable; full 2,400 steps)"
            @echo "  make train-smoke    30-step end-to-end pipeline proof"
            @echo "  make eval           Perplexity + samples from the final checkpoint"
            @echo "  make samples        Save generation samples to docs/results/"
            @echo "  make plot           Plot the training curve"
            @echo "  make serve          Start the FastAPI playground on :8000"
            @echo "  make test           Run the test suite"

collect:
            python -m ryko_lm.data.collect_corpus --source wikipedia --target-mb 19 --out data/raw/wiki.jsonl
            python -m ryko_lm.data.collect_corpus --source tashkeela --target-mb 10 --page-delay 0.4 --out data/raw/tashkeela.jsonl

clean-corpus:
            python -m ryko_lm.data.clean_corpus --inputs data/raw/wiki.jsonl data/raw/tashkeela.jsonl --out-dir data/processed

tokenizer:
            python -m ryko_lm.tokenizer.train_tokenizer --corpus data/processed/train.txt --out artifacts/tokenizer

binarize:
            python -m ryko_lm.data.binarize --tokenizer artifacts/tokenizer/ryko-tokenizer.json --in-dir data/processed --out-dir data/processed

train:
            python -m ryko_lm.training.train --config configs/models/ryko-tiny.yaml --data-dir data/processed --out-dir artifacts/training

train-smoke:
            python -m ryko_lm.tokenizer.train_tokenizer --corpus data/sample/train.txt --out artifacts/smoke/tokenizer --vocab-size 4096
            python -m ryko_lm.data.binarize --tokenizer artifacts/smoke/tokenizer/ryko-tokenizer.json --in-dir data/sample --out-dir artifacts/smoke
            python -m ryko_lm.training.train --config configs/models/ryko-tiny.yaml --data-dir artifacts/smoke --out-dir artifacts/smoke/training --total-steps 30 --eval-batches 4 --reset
            python -m ryko_lm.training.evaluate --checkpoint artifacts/smoke/training/final.pt --tokenizer artifacts/smoke/tokenizer/ryko-tokenizer.json --data-dir artifacts/smoke --max-batches 4 --max-new-tokens 24

eval:
            python -m ryko_lm.training.evaluate --checkpoint artifacts/training/final.pt

samples:
            python -m ryko_lm.training.evaluate --checkpoint artifacts/training/final.pt --out docs/results/samples.json

plot:
            python -m ryko_lm.training.plot_loss --log artifacts/training/log.jsonl --out docs/results/loss_curve.png

serve:
            uvicorn ryko_lm.serving.api:app --host 0.0.0.0 --port 8000

test:
            pytest -q

clean:
            rm -rf data/raw data/processed artifacts/smoke
            find . -name __pycache__ -type d -exec rm -rf {} +

secret-scan:
        python -m ryko_lm.security.secret_scan --tree .

knowledge-ingest:
        python -m ryko_lm.knowledge ingest --format shamela --jsonl data/raw/shamela_knowledge.jsonl --db data/knowledge/ryko.sqlite

knowledge-search:
        python -m ryko_lm.knowledge search --db data/knowledge/ryko.sqlite "$(Q)" -k 5

knowledge-cite:
        python -m ryko_lm.knowledge cite --db data/knowledge/ryko.sqlite "$(Q)"

knowledge-receipt:
        python -m ryko_lm.knowledge receipt --db data/knowledge/ryko.sqlite --out data/knowledge/ryko.receipt.json

bench-build:
        python -m ryko_lm.eval.build_bench --store data/knowledge/ryko.sqlite --out docs/benchmarks/ryko_bench_seed.jsonl

bench-run:
        python -m ryko_lm.eval.benchmarks run --benchmark docs/benchmarks/ryko_bench_seed.jsonl --checkpoint artifacts/training_v3/final.pt --tokenizer artifacts/tokenizer/ryko-tokenizer.json --store data/knowledge/ryko.sqlite --out docs/results/benchmarks_seed

fikr-collect:
        python -m ryko_lm.data.collect_fikr --manifest data/knowledge/fikr_manifest.json --out data/raw/fikr_shamela.jsonl --progress data/raw/fikr.progress.json --max-file-mb 250

fikr-build:
        python scripts/build_fikr_db.py

fikr-build-full:
        python scripts/build_fikr_db.py --no-core --db data/knowledge/ryko-full.sqlite

econom-collect:
        python -m ryko_lm.data.collect_fikr --manifest data/knowledge/econom_manifest.json --out data/raw/econom_shamela.jsonl --progress data/raw/econom.progress.json --max-file-mb 250

econom-build:
        python scripts/build_econom_db.py

econom-build-full:
        python scripts/build_econom_db.py --no-core --db data/knowledge/ryko-full.sqlite
