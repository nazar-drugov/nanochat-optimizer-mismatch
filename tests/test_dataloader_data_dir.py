"""data_dir plumbing and seeded shuffling of the pretraining/fine-tuning loader (no network)."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from nanochat import dataloader


class TinyTokenizer:
    def get_bos_token_id(self):
        return 0

    def encode(self, texts, prepend, num_threads):
        return [[prepend, *[int(ch) + 1 for ch in text]] for text in texts]


def make_dir(path, n_train_files=2, docs_per_file=24, row_group_size=4, val_docs=("9", "99")):
    """Docs are digit strings; doc k of file f is unique: str(100*f + k) padded with its length class."""
    path.mkdir(parents=True, exist_ok=True)
    all_docs = []
    for f in range(n_train_files):
        docs = [str(1000 + 100 * f + k) + "7" * (k % 5) for k in range(docs_per_file)]
        all_docs.extend(docs)
        pq.write_table(pa.table({"text": docs}), path / f"train_{f:05d}.parquet", row_group_size=row_group_size)
    pq.write_table(pa.table({"text": list(val_docs)}), path / "val_00000.parquet", row_group_size=1)
    return all_docs


def take(it, n):
    out = []
    for _ in range(n):
        x, y, state = next(it)
        out.append((x.clone(), y.clone(), state))
    return out


def doc_stream(data_dir, n, split="train", shuffle_seed=None, batch_size=2):
    gen = dataloader._document_batches(split, None, batch_size, data_dir=str(data_dir), shuffle_seed=shuffle_seed)
    docs = []
    while len(docs) < n:
        batch, _ = next(gen)
        docs.extend(batch)
    return docs[:n]


KW = dict(B=2, T=8, device="cpu", tokenizer_batch_size=3, buffer_size=4)


def test_data_dir_reads_that_directory(tmp_path):
    docs = make_dir(tmp_path / "d1")
    assert doc_stream(tmp_path / "d1", len(docs)) == docs  # None -> file order, row-group order
    assert doc_stream(tmp_path / "d1", 2, split="val") == ["9", "99"]
    it = dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(
        TinyTokenizer(), split="train", data_dir=str(tmp_path / "d1"), **KW)
    x, y, state = next(it)
    assert x.shape == (2, 8) and state["pq_idx"] == 0
    it2 = dataloader.tokenizing_distributed_data_loader_bos_bestfit(TinyTokenizer(), split="val", data_dir=str(tmp_path / "d1"), **KW)
    x, y = next(it2)
    assert set(x.flatten().tolist()) <= {0, 10}  # only BOS and '9' (=9+1) tokens from val docs


def test_default_data_dir_is_default_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    docs = make_dir(tmp_path / "base_data_fineweb_edu_100b")
    gen = dataloader._document_batches("train", None, 100, data_dir=None)
    batch, _ = next(gen)
    assert batch == docs[:4]  # first row group


def test_single_file_train_split_is_an_error(tmp_path):
    (tmp_path / "d").mkdir()
    pq.write_table(pa.table({"text": ["1"]}), tmp_path / "d" / "only.parquet")
    with pytest.raises(ValueError):
        next(dataloader._document_batches("train", None, 2, data_dir=str(tmp_path / "d")))


def test_rank_without_row_groups_fails_instead_of_hanging(tmp_path, monkeypatch):
    make_dir(tmp_path / "d", n_train_files=1, docs_per_file=4, row_group_size=4)  # 1 row group
    monkeypatch.setattr(dataloader, "get_dist_info", lambda: (True, 1, 1, 2))
    with pytest.raises(RuntimeError):
        next(dataloader._document_batches("train", None, 2, data_dir=str(tmp_path / "d")))
    with pytest.raises(RuntimeError):
        next(dataloader._document_batches("train", None, 2, data_dir=str(tmp_path / "d"), shuffle_seed=0))


def test_shuffle_none_is_unchanged_and_seeds_are_reproducible(tmp_path):
    docs = make_dir(tmp_path / "d")
    n = len(docs)
    assert doc_stream(tmp_path / "d", 2 * n, shuffle_seed=None) == docs + docs
    a = doc_stream(tmp_path / "d", 2 * n, shuffle_seed=1)
    b = doc_stream(tmp_path / "d", 2 * n, shuffle_seed=1)
    c = doc_stream(tmp_path / "d", 2 * n, shuffle_seed=2)
    assert a == b
    assert a != c
    assert a[:n] != docs
    # each epoch is a permutation of the data, and epochs are ordered differently
    assert sorted(a[:n]) == sorted(docs) and sorted(a[n:]) == sorted(docs)
    assert a[:n] != a[n:]

    def batches(seed):
        it = dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(
            TinyTokenizer(), split="train", data_dir=str(tmp_path / "d"), shuffle_seed=seed, **KW)
        return [x for x, _, _ in take(it, 12)]
    s1, s1b, s2 = batches(1), batches(1), batches(2)
    assert all(torch.equal(p, q) for p, q in zip(s1, s1b))
    assert not all(torch.equal(p, q) for p, q in zip(s1, s2))
    unshuffled = batches(None)
    assert not all(torch.equal(p, q) for p, q in zip(s1, unshuffled))


@pytest.mark.parametrize("world_size", [2, 3])
def test_shuffle_ranks_are_disjoint(tmp_path, monkeypatch, world_size):
    docs = make_dir(tmp_path / "d")  # 12 row groups of 4 docs
    per_rank = []
    for rank in range(world_size):
        monkeypatch.setattr(dataloader, "get_dist_info", lambda r=rank: (True, r, r, world_size))
        gen = dataloader._document_batches("train", None, 4, data_dir=str(tmp_path / "d"), shuffle_seed=5)
        got = []
        while True:
            batch, cursor = next(gen)
            if cursor[2] != 1:  # stop at the epoch boundary
                break
            got.extend(batch)
        per_rank.append(got)
    flat = [d for got in per_rank for d in got]
    assert len(flat) == len(set(flat)) == len(docs)
    assert sorted(flat) == sorted(docs)


@pytest.mark.parametrize("stop", [1, 5, 9])
def test_shuffled_exact_resume_reproduces_sequence(tmp_path, stop):
    make_dir(tmp_path / "d", docs_per_file=10, row_group_size=3)  # 20 docs: several epochs within 20 steps
    kw = dict(split="train", data_dir=str(tmp_path / "d"), shuffle_seed=3, **KW)
    reference = take(dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(TinyTokenizer(), **kw), 20)
    assert reference[-1][2]["epoch"] > 1  # crossed at least one epoch boundary
    saved = reference[stop - 1][2]
    assert saved["shuffle_seed"] == 3 and "next_unit_pos" in saved
    resumed = dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(TinyTokenizer(), resume_state_dict=saved, **kw)
    for x_ref, y_ref, _ in reference[stop:]:
        x, y, _ = next(resumed)
        assert torch.equal(x, x_ref) and torch.equal(y, y_ref)

    # checkpoint-style state: the prefetched batch and its state; resume replays it first
    pending_x, pending_y, state = reference[stop - 1]
    ckpt = {**state, "pending_inputs": pending_x.tolist(), "pending_targets": pending_y.tolist()}
    resumed = dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(TinyTokenizer(), resume_state_dict=ckpt, **kw)
    for x_ref, y_ref, _ in reference[stop - 1:]:
        x, y, _ = next(resumed)
        assert torch.equal(x, x_ref) and torch.equal(y, y_ref)


def test_shuffled_resume_requires_same_seed(tmp_path):
    make_dir(tmp_path / "d")
    kw = dict(split="train", data_dir=str(tmp_path / "d"), **KW)
    _, _, state = next(dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(TinyTokenizer(), shuffle_seed=3, **kw))
    with pytest.raises(ValueError):
        next(dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(TinyTokenizer(), shuffle_seed=4, resume_state_dict=state, **kw))
    with pytest.raises(ValueError):
        next(dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(TinyTokenizer(), shuffle_seed=None, resume_state_dict=state, **kw))
    _, _, plain_state = next(dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(TinyTokenizer(), **kw))
    assert "shuffle_seed" not in plain_state and "next_unit_pos" not in plain_state
    with pytest.raises(ValueError):
        next(dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(TinyTokenizer(), shuffle_seed=3, resume_state_dict=plain_state, **kw))


def test_validation_can_skip_rows_by_their_index_in_the_file(tmp_path):
    """D0 is sampled from the validation shard; validation skips those rows. Row indices
    count over the whole file, across row groups (as make_heldout records them)."""
    tmp_path.mkdir(exist_ok=True)
    docs = [f"{i:03d}" for i in range(35)]
    for name in ("train_00000.parquet", "train_00001.parquet"):
        pq.write_table(pa.table({"text": ["1"]}), tmp_path / name)
    pq.write_table(pa.table({"text": docs}), tmp_path / "val_00000.parquet", row_group_size=10)
    excluded = {0, 9, 10, 11, 23, 34}  # first/last of row groups, a whole tokenizer batch edge, the last row

    def first_pass(exclude_rows, batch_size=4):
        gen = dataloader._document_batches("val", None, batch_size, data_dir=str(tmp_path), exclude_rows=exclude_rows)
        seen = []
        while len(seen) < 35 - len(exclude_rows or ()):
            batch, cursor = next(gen)
            assert batch, "empty document batches must not be yielded"
            seen.extend(batch)
        return seen

    assert first_pass(None) == docs
    assert first_pass(excluded) == [d for i, d in enumerate(docs) if i not in excluded]
    assert first_pass({0, 1, 2, 3}, batch_size=4)[0] == "004"  # a fully excluded batch is skipped
    with pytest.raises(ValueError, match="validation split"):
        next(dataloader._document_batches("train", None, 4, data_dir=str(tmp_path), exclude_rows={1}))
    with pytest.raises(ValueError, match="fixed order"):
        next(dataloader._document_batches("val", None, 4, data_dir=str(tmp_path), shuffle_seed=0, exclude_rows={1}))
