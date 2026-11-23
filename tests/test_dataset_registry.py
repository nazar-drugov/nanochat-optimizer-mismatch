"""Pretraining dataset registry, directory resolution and data_dir plumbing (no network)."""

import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nanochat import dataset


def write_shard(path, texts, row_group_size=2):
    pq.write_table(pa.table({"text": texts}), path, row_group_size=row_group_size)


def test_registry_entries():
    fw = dataset.get_dataset_spec("fineweb-edu-100b")
    assert fw.base_url == "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
    assert fw.max_shard == 1822
    cm = dataset.get_dataset_spec("climbmix")
    assert cm.base_url == "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
    assert cm.max_shard == 6542
    assert dataset.DEFAULT_DATASET == "fineweb-edu-100b"
    assert dataset.get_dataset_spec(None) is fw
    assert dataset.index_to_filename(1822) == "shard_01822.parquet"
    with pytest.raises(ValueError):
        dataset.get_dataset_spec("nope")


def test_get_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    assert dataset.get_data_dir() == os.path.join(str(tmp_path), "base_data_fineweb_edu_100b")
    assert dataset.get_data_dir("fineweb-edu-100b") == dataset.get_data_dir()
    assert dataset.get_data_dir("climbmix") == os.path.join(str(tmp_path), "base_data_climbmix")
    assert not os.path.exists(dataset.get_data_dir())  # resolving does not create it


def test_missing_dir_raises_and_legacy_dir_is_not_used(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    legacy = tmp_path / "base_data"
    legacy.mkdir()
    write_shard(legacy / "shard_00000.parquet", ["legacy"])
    with pytest.raises(FileNotFoundError) as excinfo:
        dataset.list_parquet_files()
    assert "base_data" in str(excinfo.value)  # hint mentions the legacy dir
    assert "NOT used automatically" in str(excinfo.value)
    with pytest.raises(FileNotFoundError):
        dataset.list_parquet_files(data_dir=dataset.get_data_dir("climbmix"))


def test_list_and_iterate_custom_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    write_shard(d / "train_00001.parquet", ["c", "d"])
    write_shard(d / "train_00000.parquet", ["a", "b", "x"])
    write_shard(d / "val_00000.parquet", ["v1", "v2"])
    (d / "junk.parquet.tmp").write_text("partial download")
    paths = dataset.list_parquet_files(data_dir=str(d))
    assert [os.path.basename(p) for p in paths] == ["train_00000.parquet", "train_00001.parquet", "val_00000.parquet"]
    assert [os.path.basename(p) for p in dataset.list_split_files("val", data_dir=str(d))] == ["val_00000.parquet"]
    train_docs = [t for batch in dataset.parquets_iter_batched("train", data_dir=str(d)) for t in batch]
    assert train_docs == ["a", "b", "x", "c", "d"]
    val_docs = [t for batch in dataset.parquets_iter_batched("val", data_dir=str(d)) for t in batch]
    assert val_docs == ["v1", "v2"]
    # DDP-style striding over row groups
    rank1 = [t for batch in dataset.parquets_iter_batched("train", start=1, step=2, data_dir=str(d)) for t in batch]
    assert rank1 == ["x"]


def test_default_dir_used_when_data_dir_none(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    d = tmp_path / "base_data_fineweb_edu_100b"
    d.mkdir()
    write_shard(d / "shard_00000.parquet", ["t"])
    write_shard(d / "shard_01822.parquet", ["v"])
    assert [t for b in dataset.parquets_iter_batched("val") for t in b] == ["v"]
    assert [t for b in dataset.parquets_iter_batched("train") for t in b] == ["t"]
