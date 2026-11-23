# nanochat

![nanochat logo](dev/nanochat.png)
![scaling laws](dev/scaling_laws_jan26.png)

nanochat-optimizer-mismatch is a study of pretraining and fine-tuning with AdamW
and Muon across dense and mixture-of-experts models. It retains the nanochat
model, training, evaluation, and SFT code needed for that comparison. The
current protocol and outstanding decisions are in
[PRETRAINING_PROTOCOL.md](PRETRAINING_PROTOCOL.md) and
[CHANGES_PLAN.md](CHANGES_PLAN.md).

For questions about the repo, I recommend either using [DeepWiki](https://deepwiki.com/karpathy/nanochat) from Devin/Cognition to ask questions about the repo, or use the [Discussions tab](https://github.com/karpathy/nanochat/discussions), or come by the [#nanochat](https://discord.com/channels/1020383067459821711/1427295580895314031) channel on Discord.

## Upstream history

This study builds on nanochat. Its earlier GPT-2 speedrun results and launch
commands are archived in [dev/LEADERBOARD.md](dev/LEADERBOARD.md); they are not
instructions for the optimizer-mismatch experiments.

## Getting started

For cluster storage locations for datasets, checkpoints, and logs, see
[STORAGE_PATHS.md](STORAGE_PATHS.md). The optimizer-mismatch pretraining study
uses [runs/pretrain_study.sh](runs/pretrain_study.sh) for its pool and scratch
paths.

### Setup

nanochat uses [uv](https://docs.astral.sh/uv/) for dependency management. To install:

```bash
uv sync --extra gpu    # Use for CUDA (A100/H100/etc.)
uv sync --extra cpu    # (or) Use for CPU-only / MPS
source .venv/bin/activate
```

For development (adds pytest, matplotlib, ipykernel, transformers, etc.):

```bash
uv sync --extra gpu --group dev
```

### Pretraining study

Use [runs/pretrain_study.sh](runs/pretrain_study.sh) to prepare data, preview a
run, and launch training in a GPU allocation. It calls
[scripts/base_train.py](scripts/base_train.py) for optimization and periodic
validation. D0 is prepared by `scripts.make_heldout`; the launcher passes its
directory to training as `--val-data-dir` and checks its frozen manifest.

```bash
bash runs/pretrain_study.sh paths
RUN_NAME=moe-e8-muon-s42 bash runs/pretrain_study.sh dry-run \
    --depth=10 --ffn-type=moe --n-experts=8 --optimizer-name=muon \
    --muon-matrix-lr=0.004 --train-tokens=3000238080 \
    --core-metric-every=-1 --seed=42
```

Use `train` in place of `dry-run` after confirming the protocol and running a
short GPU test. The `train` mode logs D0 bits per byte during training.

## Validation

`scripts.base_train` computes validation bits per byte during each pretraining
run. `scripts.base_eval` is retained for standalone base-model evaluation and
for its CORE evaluator, which `base_train.py` imports. Its standalone BPB path
still needs the study's GPT-2 tokenizer and explicit validation-data path
before it can score D0 reliably. See [CHANGES_PLAN.md](CHANGES_PLAN.md).

## Precision / dtype

nanochat does not use `torch.amp.autocast`. Instead, precision is managed explicitly through a single global `COMPUTE_DTYPE` (defined in `nanochat/common.py`). By default this is auto-detected based on your hardware:

| Hardware | Default dtype | Why |
|----------|--------------|-----|
| CUDA SM 80+ (A100, H100, ...) | `bfloat16` | Native bf16 tensor cores |
| CUDA SM < 80 (V100, T4, ...) | `float32` | No bf16; fp16 available via `NANOCHAT_DTYPE=float16` (uses GradScaler) |
| CPU / MPS | `float32` | Safe default. On recent macOS, MPS also runs `NANOCHAT_DTYPE=bfloat16` fine (~25% less memory, similar speed) |

You can override the default with the `NANOCHAT_DTYPE` environment variable:

```bash
NANOCHAT_DTYPE=bfloat16 torchrun --nproc_per_node=8 -m scripts.base_train --train-tokens=125829120 --total-batch-size=524288  # force bf16
```

How it works: model weights are stored in fp32 (for optimizer precision), but our custom `Linear` layer casts them to `COMPUTE_DTYPE` during the forward pass. Embeddings are stored directly in `COMPUTE_DTYPE` to save memory. This gives us the same mixed-precision benefit as autocast but with full explicit control over what runs in which precision.

Note: `float16` training automatically enables a `GradScaler` in `base_train.py` to prevent gradient underflow. SFT supports this too. Inference in fp16 works fine everywhere.

## Guides

I've published a number of guides that might contain helpful information, most recent to least recent:

- [Feb 1 2026: Beating GPT-2 for <<$100: the nanochat journey](https://github.com/karpathy/nanochat/discussions/481)
- [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420) documents the first nanochat miniseries of models.
- To add new abilities to nanochat, see [Guide: counting r in strawberry (and how to add abilities generally)](https://github.com/karpathy/nanochat/discussions/164).
- [Oct 13 2025: original nanochat post](https://github.com/karpathy/nanochat/discussions/1) introducing nanochat, though now it contains some deprecated information and the model is a lot older (with worse results) than current master.

## File structure

```
.
├── LICENSE
├── README.md
├── dev
│   ├── nanochat.png
│   └── repackage_data_reference.py # Pretraining data shard generation
├── nanochat
│   ├── __init__.py                 # empty
│   ├── checkpoint_manager.py       # Save/Load model checkpoints
│   ├── common.py                   # Misc small utilities, quality of life
│   ├── core_eval.py                # Evaluates base model CORE score (DCLM paper)
│   ├── dataloader.py               # Tokenizing Distributed Data Loader
│   ├── dataset.py                  # Download/read utils for pretraining data
│   ├── engine.py                   # Efficient model inference with KV Cache
│   ├── execution.py                # Allows the LLM to execute Python code as tool
│   ├── gpt.py                      # The GPT nn.Module Transformer
│   ├── loss_eval.py                # Evaluate bits per byte (instead of loss)
│   ├── optim.py                    # AdamW + Muon optimizer, 1GPU and distributed
│   └── tokenizer.py                # BPE Tokenizer wrapper in style of GPT-4
├── pyproject.toml
├── runs
│   └── pretrain_study.sh            # Cluster data preparation and pretraining launcher
├── scripts
│   ├── base_eval.py                # Base model: CORE score, bits per byte, samples
│   ├── base_train.py               # Base model: train and periodic validation
│   ├── chat_eval.py                # SFT model: eval tasks
│   ├── chat_sft.py                 # SFT model: train
│   ├── count_tokens_chinchilla.py  # Optional token budget calculator
│   └── make_heldout.py             # Build D0 and scan training shards
├── tasks
│   ├── arc.py                      # Multiple choice science questions
│   ├── common.py                   # TaskMixture | TaskSequence
│   ├── gsm8k.py                    # 8K Grade School Math questions
│   ├── humaneval.py                # Misnomer; Simple Python coding task
│   ├── mmlu.py                     # Multiple choice questions, broad topics
│   └── smoltalk.py                 # Conglomerate dataset of SmolTalk from HF
├── tests
│   ├── test_attention_fallback.py  # FA3/SDPA attention fallback
│   ├── test_engine.py              # Inference engine, KV cache
│   ├── test_execution.py           # Sandboxed code execution
│   ├── test_optim.py               # Distributed optimizer (needs GPU)
│   ├── test_tasks.py               # Task slicing, mixtures, HubDataset
│   └── test_tokenizer.py           # BPE round-trips, chat rendering
└── uv.lock
```

## Contributing

The goal of nanochat is to improve the state of the art in micro models that are accessible to work with end to end on budgets of < $1000 dollars. Accessibility is about overall cost but also about cognitive complexity - nanochat is not an exhaustively configurable LLM "framework"; there are no giant configuration objects, model factories, or if-then-else monsters in the code base. It is a single, cohesive, minimal, readable, hackable, maximally-forkable "strong baseline" codebase designed to run start to end and produce a ChatGPT model you can talk to. Currently, the most interesting part personally is speeding up the latency to GPT-2 (i.e. getting a CORE score above 0.256525). Currently this takes ~1.5 hours (down from 3h), but by improving the pretraining stage we can improve this further.

Current AI policy: disclosure. When submitting a PR, please declare any parts that had substantial LLM contribution and that you have not written or that you do not fully understand.

## Acknowledgements

- The name (nanochat) derives from my earlier project [nanoGPT](https://github.com/karpathy/nanoGPT), which only covered pretraining.
- nanochat is also inspired by [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt), which gamified the nanoGPT repo with clear metrics and a leaderboard, and borrows a lot of its ideas and some implementation for pretraining.
- Thank you to [HuggingFace](https://huggingface.co/) for fineweb and smoltalk.
- Thank you [Lambda](https://lambda.ai/service/gpu-cloud) for the compute used in developing this project.
- Thank you to chief LLM whisperer 🧙‍♂️ Alec Radford for advice/guidance.
- Thank you to the repo czar Sofie [@svlandeg](https://github.com/svlandeg) for help with managing issues, pull requests and discussions of nanochat.

## Cite

If you find nanochat helpful in your research cite simply as:

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## License

MIT
