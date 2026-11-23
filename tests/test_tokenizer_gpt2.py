"""GPT-2 tokenizer support (team decision 2026-11-23). Skipped if tiktoken cannot
load the GPT-2 files (they are downloaded and cached on first use)."""

import pytest
import torch

from nanochat.engine import Engine
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import TOKENIZERS, compute_token_bytes, get_token_bytes, get_tokenizer


@pytest.fixture(scope="module")
def gpt2():
    try:
        return get_tokenizer("gpt2")
    except Exception as exc:  # no network and no cache
        pytest.skip(f"GPT-2 tokenizer unavailable: {exc}")


def test_gpt2_vocab_and_bos(gpt2):
    assert "gpt2" in TOKENIZERS
    assert gpt2.get_vocab_size() == 50257
    assert gpt2.get_bos_token_id() == 50256  # <|endoftext|>
    ids = gpt2.encode(["hello world", "def f(x):\n    return {x}"], prepend=gpt2.get_bos_token_id())
    assert all(row[0] == 50256 for row in ids)
    assert gpt2.decode(ids[1][1:]) == "def f(x):\n    return {x}"


@pytest.mark.parametrize("text", ["Hello world!", "    def f():\n\treturn {'a': 1}", "naïve café — 你好 🌍", ""])
def test_token_bytes_sum_to_utf8_length(gpt2, text):
    token_bytes = compute_token_bytes(gpt2)
    assert token_bytes.shape == (50257,) and token_bytes[50256] == 0
    ids = gpt2.encode(text)
    assert int(token_bytes[ids].sum()) == len(text.encode("utf-8"))


def test_get_token_bytes_by_name(gpt2):
    torch.testing.assert_close(get_token_bytes("cpu", name="gpt2"), compute_token_bytes(gpt2))
    with pytest.raises(ValueError):
        get_tokenizer("gpt4")


def test_engine_samples_with_a_tokenizer_without_chat_tokens(gpt2):
    model = GPT(GPTConfig(sequence_len=32, vocab_size=50257, n_layer=1, n_head=1, n_kv_head=1, n_embd=16,
                          ffn_type="moe", n_experts=2))
    model.init_weights()
    model.eval()
    prompt = gpt2.encode("The capital of France is", prepend=gpt2.get_bos_token_id())
    samples, masks = Engine(model, gpt2).generate_batch(prompt, num_samples=2, max_tokens=4, temperature=0)
    assert len(samples) == 2 and all(len(s) <= len(prompt) + 4 for s in samples)
    assert samples[0][:len(prompt)] == prompt
