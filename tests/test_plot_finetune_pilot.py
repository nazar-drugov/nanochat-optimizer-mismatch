"""Smoke test: the pilot chart renders from a summary list, including a diverged cell."""
import json

from scripts import plot_finetune_pilot as p


def test_pilot_chart_renders_and_marks_diverged_cells(tmp_path):
    cells = [{"pre": pre, "ft": ft, "T": T, "lr": lr, "n": 3, "learn": learn, "forget": .1 * T / 60 + lr * 100,
              "d1": 2 - learn}
             for pre, ft in p.PAIRS for T in (60, 240) for lr, learn in ((1e-4, .8), (2e-4, 1.0), (4e-4, -1.5))]
    summary = tmp_path / "finetune_pilot_data_summary.json"
    summary.write_text(json.dumps(cells))
    assert p.selected(cells)[("adamw", "muon", 60)] == 2e-4
    out = p.plot(summary, tmp_path / "figs")
    assert out.name == "finetune_pilot_data_lr.png" and out.stat().st_size > 10_000
