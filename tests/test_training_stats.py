"""Numerical and schema checks for pretraining diagnostics."""

from types import SimpleNamespace
import ast
import json
from pathlib import Path

import torch
from torch import nn

from nanochat.training_stats import (
    accumulate_router_stats, add_router_log_fields, expert_gradient_squared_norms,
    write_train_stats_jsonl,
)


def test_two_equal_microbatches_and_exact_moe_log_keys():
    def layer(shares, probs, entropy, z_loss):
        return {
            "expert_shares": torch.tensor(shares),
            "mean_expert_probs": torch.tensor(probs),
            "router_entropy": torch.tensor(entropy),
            "z_loss": torch.tensor(z_loss),
        }

    first = [layer([1, 0, 0, 0], [.7, .1, .1, .1], .8, 2),
             layer([0, 1, 0, 0], [.1, .7, .1, .1], .9, 3)]
    second = [layer([0, .5, .5, 0], [.1, .4, .4, .1], 1.2, 4),
              layer([.5, 0, 0, .5], [.4, .1, .1, .4], 1.3, 5)]
    sums = accumulate_router_stats(None, first, 2)
    sums = accumulate_router_stats(sums, second, 2)
    torch.testing.assert_close(sums["expert_shares"], torch.tensor([[.5, .25, .25, 0], [.25, .5, 0, .25]]))
    torch.testing.assert_close(sums["mean_expert_probs"], torch.tensor([[.4, .25, .25, .1], [.25, .4, .1, .25]]))
    torch.testing.assert_close(sums["router_entropy"], torch.tensor([1., 1.1]))
    torch.testing.assert_close(sums["z_loss"], torch.tensor([3., 4.]))

    record = add_router_log_fields({"step": 0}, sums, torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]]))
    expected = {"step"}
    for layer_idx in range(2):
        prefix = f"train/moe/layer_{layer_idx}/"
        expected.update({prefix + "router_entropy", prefix + "z_loss"})
        for expert_idx in range(4):
            expected.update({prefix + f"expert_{expert_idx}_{suffix}"
                             for suffix in ("share", "mean_prob", "grad_norm")})
    assert set(record) == expected
    assert record["train/moe/layer_0/expert_3_share"] == 0
    assert record["train/moe/layer_0/expert_3_mean_prob"] > 0


def test_expert_gradient_norm_from_known_parameter_gradients():
    experts = nn.ModuleList([nn.Linear(2, 2, bias=True) for _ in range(4)])
    model = SimpleNamespace(
        config=SimpleNamespace(n_layer=1, n_experts=4),
        transformer=SimpleNamespace(h=[SimpleNamespace(mlp=SimpleNamespace(experts=experts))]),
    )
    experts[0].weight.grad = torch.tensor([[3., 4.], [0., 0.]])
    experts[0].bias.grad = torch.tensor([12., 0.])
    experts[1].weight.grad = torch.tensor([[1., 2.], [2., 4.]])
    squared = expert_gradient_squared_norms(model, torch.device("cpu"))
    torch.testing.assert_close(squared.sqrt(), torch.tensor([[13., 5., 0., 0.]]))


def test_logging_defaults_and_exact_jsonl_record(tmp_path):
    source = Path(__file__).resolve().parents[1] / "scripts" / "base_train.py"
    tree = ast.parse(source.read_text())
    defaults = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            if node.args and isinstance(node.args[0], ast.Constant):
                for keyword in node.keywords:
                    if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                        defaults[node.args[0].value] = keyword.value.value
    assert defaults["--train-stats-every"] == 100
    assert defaults["--train-stats-jsonl"] is None

    path = tmp_path / "nested" / "stats.jsonl"
    record = {"step": 0, "train/moe/layer_0/expert_0_share": .25}
    write_train_stats_jsonl(path, record, False)
    assert not path.exists()
    write_train_stats_jsonl(path, record, True)
    assert [json.loads(line) for line in path.read_text().splitlines()] == [record]
