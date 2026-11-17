"""Resume must preserve the next packed training batch, not only its row group."""

import pyarrow as pa
import pyarrow.parquet as pq
import torch


class TinyTokenizer:
    def get_bos_token_id(self):
        return 0

    def encode(self, texts, prepend, num_threads):
        return [[prepend, *[int(ch) + 1 for ch in text]] for text in texts]


def test_exact_resume_replays_next_packed_batches(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    from nanochat import dataloader

    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    texts = [str(i % 10) * (1 + i % 5) for i in range(30)]
    pq.write_table(pa.table({"text": texts}), train, row_group_size=10)
    pq.write_table(pa.table({"text": ["1"]}), val)
    monkeypatch.setattr(dataloader, "list_parquet_files", lambda **_kwargs: [str(train), str(val)])
    kwargs = dict(B=1, T=8, split="train", device="cpu", tokenizer_batch_size=3, buffer_size=4)
    original = dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(TinyTokenizer(), **kwargs)
    for _ in range(3):
        pending_x, pending_y, saved_state = next(original)
    pending_x, pending_y = pending_x.clone(), pending_y.clone()
    expected = []
    for _ in range(3):
        x, y, _ = next(original)
        expected.append((x.clone(), y.clone()))

    resumed = dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(
        TinyTokenizer(), resume_state_dict=saved_state, **kwargs,
    )
    assert saved_state["doc_buffer"]
    for expected_x, expected_y in expected:
        x, y, _ = next(resumed)
        assert torch.equal(x, expected_x)
        assert torch.equal(y, expected_y)

    # Training checkpoints after prefetching this batch, so resume must replay it.
    checkpoint_state = {
        **saved_state,
        "pending_inputs": pending_x.tolist(),
        "pending_targets": pending_y.tolist(),
    }
    checkpoint_resume = dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(
        TinyTokenizer(), resume_state_dict=checkpoint_state, **kwargs,
    )
    x, y, _ = next(checkpoint_resume)
    assert torch.equal(x, pending_x)
    assert torch.equal(y, pending_y)
    x, y, _ = next(checkpoint_resume)
    assert torch.equal(x, expected[0][0])
    assert torch.equal(y, expected[0][1])


def test_legacy_row_group_state_still_loads(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    from nanochat import dataloader

    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    pq.write_table(pa.table({"text": ["123"] * 10}), train, row_group_size=5)
    pq.write_table(pa.table({"text": ["1"]}), val)
    monkeypatch.setattr(dataloader, "list_parquet_files", lambda **_kwargs: [str(train), str(val)])
    iterator = dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(
        TinyTokenizer(), 1, 4, split="train", device="cpu", buffer_size=2,
        resume_state_dict={"pq_idx": 0, "rg_idx": 0, "epoch": 1},
    )
    _, _, state = next(iterator)
    assert state["rg_idx"] == 1
