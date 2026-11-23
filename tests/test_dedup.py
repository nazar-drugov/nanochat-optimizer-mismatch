"""MinHash/LSH near-duplicate detection, n-gram overlap, and the D0 / overlap scripts (no network)."""

import json
import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nanochat import dedup


def random_doc(rng, vocab, n_words=200):
    return " ".join(rng.choice(vocab) for _ in range(n_words))


def edit_words(rng, text, n_edits, vocab):
    words = text.split()
    for i in rng.sample(range(len(words)), n_edits):
        words[i] = rng.choice(vocab)
    return " ".join(words)


@pytest.fixture
def corpus():
    rng = random.Random(0)
    vocab = [f"w{i}" for i in range(3000)]
    docs = [random_doc(rng, vocab) for _ in range(30)]
    return rng, vocab, docs


def test_word_hashes_normalization_and_determinism():
    h1, c1 = dedup.word_hashes(["Hello, World!", "", None, "x  y"])
    assert c1.tolist() == [2, 0, 0, 2]
    h2, _ = dedup.word_hashes(["hello world"])
    assert np.array_equal(h1[:2], h2)  # case and punctuation insensitive
    h3, c3 = dedup.word_hashes(["Hello, World!"], normalize=False)
    assert c3.tolist() == [2] and not np.array_equal(h3, h2)
    assert np.array_equal(dedup.word_hashes(["über straße"])[0], dedup.word_hashes(["ÜBER straße"])[0])


def test_ngrams_do_not_cross_documents():
    wh, counts = dedup.word_hashes(["a b c", "d e f g", "h i"])
    h, d = dedup.ngram_hashes(wh, counts, 3)
    assert d.tolist() == [0, 1, 1]  # 1 trigram in doc 0, 2 in doc 1, none in doc 2
    # same n-gram in two docs hashes identically; different order differs
    wh, counts = dedup.word_hashes(["p q r", "x p q r", "r q p"])
    h, d = dedup.ngram_hashes(wh, counts, 3)
    assert h[0] == h[2] and h[0] != h[3]


def test_minhash_estimates_jaccard(corpus):
    rng, vocab, docs = corpus
    a, b = docs[0], edit_words(rng, docs[0], 10, vocab)
    hashes, doc_ids, n = dedup.doc_shingles([a, b, docs[1]])
    sets = [set(hashes[doc_ids == i].tolist()) for i in range(n)]
    true_j = len(sets[0] & sets[1]) / len(sets[0] | sets[1])
    sig = dedup.MinHasher(num_perm=256, seed=3).signatures(hashes, doc_ids, n)
    assert abs(dedup.estimated_jaccard(sig[0], sig[1]) - true_j) < 0.1
    assert dedup.estimated_jaccard(sig[0], sig[2]) < 0.05


def test_lsh_parameters():
    assert dedup.NUM_PERM == 128 and dedup.LSH_BANDS * dedup.LSH_ROWS == 128
    assert abs(dedup.lsh_threshold(16, 8) - 0.7071) < 1e-3
    assert dedup.lsh_collision_probability(0.8, 16, 8) > 0.94
    assert dedup.lsh_collision_probability(0.3, 16, 8) < 0.002


@pytest.mark.parametrize("prefilter_margin", [dedup.PREFILTER_MARGIN, None])
def test_near_duplicates_found_unrelated_kept(corpus, prefilter_margin):
    rng, vocab, docs = corpus
    index = dedup.NearDupIndex(docs, prefilter_margin=prefilter_margin)
    stream = [
        random_doc(rng, vocab),                              # 0 unrelated
        edit_words(rng, docs[3], 1, vocab),                  # 1 near-dup of 3
        docs[7].upper().replace(" ", "  "),                  # 2 same words, different case/spacing
        docs[11],                                            # 3 exact dup
        edit_words(rng, docs[12], 100, vocab),               # 4 half the words changed: not a near-dup
        " ".join(docs[20].split()[:100]),                    # 5 first half of a doc: Jaccard ~0.5
        "",                                                  # 6 empty
        "too short",                                         # 7 fewer words than a shingle
    ]
    matches, stats = index.query_texts(stream)
    found = {(q, j) for q, j, _ in matches}
    assert found == {(1, 3), (2, 7), (3, 11)}
    assert stats["docs"] == len(stream)
    assert all(est >= dedup.JACCARD_THRESHOLD for _, _, est in matches)


def test_internal_duplicates(corpus):
    rng, vocab, docs = corpus
    index = dedup.NearDupIndex(docs[:5] + [edit_words(rng, docs[2], 1, vocab), "x"])
    pairs = index.internal_duplicates()
    assert [(i, j) for i, j, _ in pairs] == [(5, 2)]
    assert index.shingle_counts[6] == 0


def test_ngram_overlap_counts():
    rng = random.Random(1)
    vocab = [f"v{i}" for i in range(5000)]
    ref = [random_doc(rng, vocab, 50) for _ in range(4)]   # 38 distinct 13-grams each
    ov = dedup.NgramOverlap(ref, n=13)
    assert ov.summarize([])["frac_unique_ngrams_overlapping"] == 0.0
    words = ref[1].split()
    stream = [
        "prefix " + " ".join(words[10:30]) + " suffix",    # 20 consecutive words -> 8 13-grams of doc 1
        random_doc(rng, vocab, 300),
        " ".join(ref[3].split()[:13]).upper() + "!",        # exactly one 13-gram of doc 3, different case
    ]
    matched = np.unique(np.concatenate([ov.scan(stream[:2]), ov.scan(stream[2:])]))
    s = ov.summarize(matched)
    assert s["reference_unique_ngrams"] == 4 * 38
    assert s["overlapping_unique_ngrams"] == 9
    assert s["docs_with_overlap"] == 2 and s["frac_docs_with_overlap"] == 0.5
    assert abs(s["frac_unique_ngrams_overlapping"] - 9 / 152) < 1e-12
    raw = dedup.NgramOverlap(ref, n=13, normalize=False)
    assert len(raw.scan(stream[2:])) == 0  # case differs: no match without normalization


def write_shard(path, texts, row_group_size=8):
    pq.write_table(pa.table({"text": texts}), path, row_group_size=row_group_size)


def test_make_heldout_end_to_end(tmp_path, corpus):
    from scripts.make_heldout import build_heldout
    rng, vocab, _ = corpus
    val_docs = [random_doc(rng, vocab, 100) for _ in range(40)]
    val_docs[5] = ""  # empty rows are never sampled
    train0 = [random_doc(rng, vocab, 100) for _ in range(30)]
    train1 = [random_doc(rng, vocab, 100) for _ in range(30)]
    planted = [2, 9, 17, 33]  # near-dups of these val rows live in the training shards
    train0[4] = edit_words(rng, val_docs[2], 1, vocab)
    train0[20] = val_docs[9]
    train1[0] = edit_words(rng, val_docs[17], 2, vocab)
    train1[29] = val_docs[33].upper()
    data = tmp_path / "data"
    data.mkdir()
    write_shard(data / "shard_00000.parquet", train0)
    write_shard(data / "shard_00001.parquet", train1)
    write_shard(data / "shard_00002.parquet", val_docs)
    out = tmp_path / "out"
    # target = all non-empty val docs, so every planted doc is a candidate
    total_tokens = sum(-(-len(t) // 4) for t in val_docs)
    report = build_heldout(str(data), str(out), "synthetic", num_tokens=total_tokens, oversample=0.0,
                           token_count="chars", workers=1, seed=0)
    dropped = sorted(d["row_index"] for d in report["dropped_near_dup_in_training"])
    assert dropped == planted
    assert report["train_shards_scanned"]["files"] == ["shard_00000.parquet", "shard_00001.parquet"]
    assert report["train_shards_scanned"]["docs"] == 60
    table = pq.read_table(out / "d0.parquet")
    assert table.column_names == ["text", "source_shard", "row_index"]
    rows = table.column("row_index").to_pylist()
    assert set(rows).isdisjoint(planted) and 5 not in rows
    assert len(rows) == 39 - len(planted) == report["kept"]["docs"]
    assert set(table.column("source_shard").to_pylist()) == {"shard_00002.parquet"}
    for text, row in zip(table.column("text").to_pylist(), rows):  # doc IDs point at the right val rows
        assert val_docs[row] == text
    assert json.loads((out / "d0_report.json").read_text())["dropped"]["near_dup_in_training"] == 4

    # seeded sampling is reproducible and a smaller target keeps a prefix of the sample order
    report2 = build_heldout(str(data), str(tmp_path / "out2"), "synthetic", num_tokens=200, oversample=0.0,
                            token_count="chars", workers=1, seed=0, max_train_shards=1)
    assert report2["train_shards_scanned"]["files"] == ["shard_00000.parquet"]
    assert report2["kept"]["tokens"] >= 200

# test_ngram_overlap_script lives with scripts/ngram_overlap.py (fine-tuning data, parked).
