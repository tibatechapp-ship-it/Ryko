"""Data-pipeline tests: streaming cleaning + dedup logic on synthetic Arabic text."""

import json

import numpy as np

from ryko_lm.data.clean_corpus import (
    CleanOptions,
    _fingerprint,
    _fingerprint64,
    _index_documents,
    clean_streaming,
    normalize_arabic,
    split_streaming,
)
from ryko_lm.data.binarize import _encode_file
from ryko_lm.tokenizer.loader import RykoTokenizer

from conftest import TOKENIZER_JSON, requires_tokenizer


def _opts(**kw):
    base = dict(
        min_line_len=10, min_arabic_letters=5, min_arabic_ratio=0.5,
        remove_diacritics=True, unify_alef=False, unify_yaa=False,
        taa_marbuta_to_haa=False,
    )
    base.update(kw)
    return CleanOptions(**base)


def test_fingerprint_ignores_punctuation_and_spaces():
    a = _fingerprint("مرحبا بالعالم!")
    b = _fingerprint("مرحبا  بالعالم")
    assert a == b
    # int64 fingerprint must agree with the hex version
    assert _fingerprint64("مرحبا بالعالم!") == int.from_bytes(
        bytes.fromhex(_fingerprint("مرحبا  بالعالم"))[:8], "big"
    )


def test_normalize_arabic_pipeline_semantics():
    assert normalize_arabic("وَالْكِتَابُ", remove_diacritics=True) == "والكتاب"
    assert normalize_arabic("\u0640\u0640\u0640", remove_diacritics=True).strip() == ""
    assert "أ" in normalize_arabic("أكل", unify_alef=False)
    assert "أ" not in normalize_arabic("أكل", unify_alef=True)


def test_clean_streaming_dedups_and_reports(tmp_path):
    inputs = []
    for i, docs in enumerate([
        ["اللغة العربية أداة الفكر الأولى.\nسطر مميز ثانٍ في الوثيقة.", "سطر مكرر قديم آخر هنا."],
        ["سطر مكرر قديم آخر هنا.", "وثيقة ثانية مختلفة تماما عن سابقتها."],
    ]):
        p = tmp_path / f"in{i}.jsonl"
        p.write_text(
            "\n".join(json.dumps({"id": f"d{i}-{j}", "text": t}) for j, t in enumerate(docs)),
            encoding="utf-8",
        )
        inputs.append(p)

    tmp_all = tmp_path / "all.tmp.txt"
    stats = clean_streaming(inputs, _opts(), tmp_all)
    assert stats["mode"] == "streaming"
    # one paragraph deduplicated across files; the second document of file 1
    # consisted of ONLY that paragraph, so it disappears entirely (legacy
    # semantics: docs with zero unique paragraphs are not emitted)
    assert stats["paragraphs_dropped_duplicate"] == 1
    assert stats["documents_total"] == 3
    text = tmp_all.read_text(encoding="utf-8")
    docs = [d for d in text.split("\n\n") if d.strip()]
    assert len(docs) == 3


def test_split_streaming_is_seeded_and_partitioning(tmp_path):
    tmp = tmp_path / "all.tmp.txt"
    docs = [f"وثيقة رقم {i} تحتوي سطرا واحدا طويلا بما يكفي للفلتر." for i in range(100)]
    tmp.write_text("\n\n".join(docs) + "\n\n", encoding="utf-8")

    train_a, val_a = tmp_path / "ta.txt", tmp_path / "va.txt"
    s1 = split_streaming(tmp, train_a, val_a, val_ratio=0.1, seed=1337)
    assert s1["documents_train"] + s1["documents_val"] == 100
    assert s1["documents_val"] == 10

    # deterministic: same seed -> identical split
    train_b, val_b = tmp_path / "tb.txt", tmp_path / "vb.txt"
    s2 = split_streaming(tmp, train_b, val_b, val_ratio=0.1, seed=1337)
    assert train_a.read_text(encoding="utf-8") == train_b.read_text(encoding="utf-8")
    assert val_a.read_text(encoding="utf-8") == val_b.read_text(encoding="utf-8")

    # partition: no doc in both splits
    va = {l for l in val_a.read_text(encoding="utf-8").splitlines() if l.strip()}
    ta = {l for l in train_a.read_text(encoding="utf-8").splitlines() if l.strip()}
    assert not (va & ta)


def test_index_documents_byte_offsets(tmp_path):
    tmp = tmp_path / "docs.txt"
    tmp.write_text("أول وثيقة هنا\n\nوثانية\nسطران\n\nثالثة\n", encoding="utf-8")
    offs = _index_documents(tmp)
    assert len(offs) == 3
    with tmp.open("rb") as f:
        first = f.seek(offs[0][0]) or f.read(offs[0][1] - offs[0][0]).decode("utf-8")
    assert first.startswith("أول")


@requires_tokenizer
def test_binarize_file_roundtrip(tmp_path):
    tok = RykoTokenizer.from_file(TOKENIZER_JSON)
    src = tmp_path / "train.txt"
    src.write_text("اللغة العربية جميلة\n\nوثانية تماما\n", encoding="utf-8")
    out = tmp_path / "train.bin"
    n_tokens = _encode_file(tok, src, out)
    arr = np.fromfile(out, dtype=np.uint16)
    assert n_tokens == arr.size
    assert arr.size >= 8
    # first paragraph starts with BOS, ends with EOS
    assert arr[0] == tok.bos_id
    decoded = tok.decode(arr.tolist())
    assert "اللغة" in decoded


# ---------------------------------------------------------------------------
# augment_corpus: copy mode vs in-place mode (byte-identical outputs)
# ---------------------------------------------------------------------------
from ryko_lm.data.augment_corpus import main as augment_main  # noqa: E402


def _raw_jsonl(path, docs):
    with path.open("w", encoding="utf-8") as out:
        for i, text in enumerate(docs):
            out.write(json.dumps({"id": f"doc-{i:04d}", "text": text},
                                 ensure_ascii=False) + "\n")


_AR = "في البداية كان الكلام محفوظا في الصدور ولا كتابة"


def test_augment_copy_mode_concatenates_old_and_new(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "train.txt").write_text("فقرة قديمة أولى قابلة للفهرسة تماما\n\n"
                                        "فقرة قديمة ثانية مختلفة عن الأولى\n",
                                        encoding="utf-8")
    (existing / "val.txt").write_text("", encoding="utf-8")
    raw = tmp_path / "new.jsonl"
    _raw_jsonl(raw, [_AR + " ومعه جملة جديدة طويلة للتنظيف والتوسيع"])
    out = tmp_path / "aug"
    rc = augment_main(["--existing-dir", str(existing), "--inputs", str(raw),
                       "--out-dir", str(out)])
    assert rc == 0
    assert "فقرة قديمة أولى" in (out / "train.txt").read_text(encoding="utf-8")
    assert _AR.split()[0] in (out / "train.txt").read_text(encoding="utf-8")
    stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    assert stats["mode"] == "copy"
    assert stats["new_sources"]["new.jsonl"]["docs"] == 1


def test_augment_in_place_appends_without_copy_and_matches_copy_mode(tmp_path):
    existing = tmp_path / "corpus"
    existing.mkdir()
    old_train = "فقرة قديمة أولى قابلة للفهرسة تماما\n\nفقرة قديمة ثانية مختلفة عن الأولى\n"
    (existing / "train.txt").write_text(old_train, encoding="utf-8")
    (existing / "val.txt").write_text("", encoding="utf-8")
    raw = tmp_path / "new.jsonl"
    _raw_jsonl(raw, [_AR + " ومعه جملة جديدة طويلة للتنظيف والتوسيع"])

    rc = augment_main(["--existing-dir", str(existing), "--inputs", str(raw),
                       "--out-dir", str(existing), "--in-place"])
    assert rc == 0
    merged = (existing / "train.txt").read_text(encoding="utf-8")
    assert "فقرة قديمة أولى" in merged          # old content intact
    assert _AR.split()[0] in merged             # new content appended
    assert not (existing / "train_new.txt").exists()  # temp removed
    stats = json.loads((existing / "stats.json").read_text(encoding="utf-8"))
    assert stats["mode"] == "in-place"

    # in-place result must equal the copy-mode result for the same inputs
    ref_old = tmp_path / "ref_old"
    ref_old.mkdir()
    (ref_old / "train.txt").write_text(old_train, encoding="utf-8")
    (ref_old / "val.txt").write_text("", encoding="utf-8")
    ref_out = tmp_path / "ref_out"
    augment_main(["--existing-dir", str(ref_old), "--inputs", str(raw),
                  "--out-dir", str(ref_out)])
    assert merged == (ref_out / "train.txt").read_text(encoding="utf-8")


def test_augment_in_place_requires_existing_files(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    raw = tmp_path / "new.jsonl"
    _raw_jsonl(raw, [_AR])
    rc = augment_main(["--existing-dir", str(empty), "--inputs", str(raw),
                       "--out-dir", str(empty), "--in-place"])
    assert rc == 2
