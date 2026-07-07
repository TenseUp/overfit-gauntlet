"""Tests for the ``gauntlet`` console script (run / lint subcommands)."""
from __future__ import annotations

import os

import numpy as np
import pytest

from overfit_gauntlet.cli import main


def _write_csv(path, values, header=None, costs=None, cost_header="cost"):
    lines = []
    if header is not None:
        cols = [header] + ([cost_header] if costs is not None else [])
        lines.append(",".join(cols))
    for i, v in enumerate(values):
        row = [repr(float(v))]
        if costs is not None:
            row.append(repr(float(costs[i])))
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n")


def test_run_validated_exits_zero(tmp_path, edge_returns, capsys):
    csv = tmp_path / "edge.csv"
    _write_csv(csv, edge_returns, header="returns")
    code = main(["run", str(csv), "--col", "returns"])
    out = capsys.readouterr().out
    assert code == 0
    assert "**Verdict: VALIDATED**" in out


def test_run_noise_exits_one(tmp_path, noise_returns, capsys):
    csv = tmp_path / "noise.csv"
    _write_csv(csv, noise_returns, header="returns")
    code = main(["run", str(csv), "--col", "returns"])
    assert code == 1  # CI gate: not VALIDATED
    assert "Verdict:" in capsys.readouterr().out


def test_run_headerless_csv(tmp_path, edge_returns):
    csv = tmp_path / "edge_nohdr.csv"
    _write_csv(csv, edge_returns, header=None)
    code = main(["run", str(csv)])
    assert code == 0


def test_run_with_costs_column(tmp_path, edge_returns, capsys):
    csv = tmp_path / "edge_costs.csv"
    costs = np.full(len(edge_returns), 0.0005)
    _write_csv(csv, edge_returns, header="returns", costs=costs)
    code = main(["run", str(csv), "--col", "returns", "--costs-col", "cost"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Cost stress" in out


def test_run_missing_file_is_usage_error(tmp_path):
    code = main(["run", str(tmp_path / "nope.csv")])
    assert code == 2


def test_lint_leaky_fixture_exits_one(fixtures_dir, capsys):
    code = main(["lint", os.path.join(fixtures_dir, "leaky.py")])
    out = capsys.readouterr().out
    assert code == 1  # error-severity findings present
    assert "LA01" in out


def test_lint_clean_fixture_exits_zero(fixtures_dir, capsys):
    code = main(["lint", os.path.join(fixtures_dir, "clean.py")])
    out = capsys.readouterr().out
    assert code == 0
    assert "No findings" in out


def test_version_flag_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "0.1.0" in capsys.readouterr().out
