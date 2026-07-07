"""Tests for the AST look-ahead linter, driven by fixture files.

``tests/fixtures/leaky.py`` deliberately triggers every rule LA01-LA08; the
sibling ``clean.py`` is causal and must yield zero findings.
"""
from __future__ import annotations

import os

import pytest

from overfit_gauntlet.lint import Finding, lint_file, lint_path, lint_source

ALL_CODES = {"LA01", "LA02", "LA03", "LA04", "LA05", "LA06", "LA07", "LA08"}


def test_leaky_fixture_fires_every_rule(fixtures_dir):
    findings = lint_file(os.path.join(fixtures_dir, "leaky.py"))
    codes = {f.code for f in findings}
    assert ALL_CODES <= codes, f"missing: {ALL_CODES - codes}"


def test_clean_fixture_has_zero_findings(fixtures_dir):
    assert lint_file(os.path.join(fixtures_dir, "clean.py")) == []


def test_lint_path_recurses_directory(fixtures_dir):
    findings = lint_path(fixtures_dir)
    # only the leaky fixture contributes findings; every finding carries a path
    assert findings
    assert all(f.file.endswith(".py") for f in findings)
    assert {f.code for f in findings} >= ALL_CODES
    # findings are sorted by (file, line, code)
    assert findings == sorted(findings, key=lambda f: (f.file, f.line, f.code))


@pytest.mark.parametrize(
    "src, code",
    [
        ("y = df['p'].shift(-1)", "LA01"),
        ("y = df['p'].shift(periods=-2)", "LA01"),
        ("y = df['p'].rolling(5, center=True).mean()", "LA02"),
        ("y = df['p'].bfill()", "LA03"),
        ("y = df['p'].fillna(method='bfill')", "LA03"),
        ("a, b = train_test_split(X, y)", "LA04"),
        ("cv = KFold(n_splits=5, shuffle=True)", "LA05"),
        ("z = (x - x.mean()) / x.std()", "LA07"),
        ("y = df['p'].pct_change(-3)", "LA08"),
    ],
)
def test_each_rule_in_isolation(src, code):
    codes = {f.code for f in lint_source(src)}
    assert code in codes


def test_severity_split():
    # LA07 (full-sample normalization) and plain KFold are warnings, not errors
    warn = lint_source("z = (x - x.mean()) / x.std()")
    assert warn and all(f.severity == "warn" for f in warn if f.code == "LA07")
    plain_kfold = lint_source("cv = KFold(n_splits=5)")
    assert any(f.code == "LA05" and f.severity == "warn" for f in plain_kfold)


def test_shuffle_false_is_not_flagged():
    assert lint_source("a, b = train_test_split(X, y, shuffle=False)") == []


def test_positive_shift_is_clean():
    assert lint_source("y = df['p'].shift(1)") == []
    assert lint_source("y = df['p'].pct_change(1)") == []


def test_finding_repr_and_fields():
    f = lint_source("y = df['p'].shift(-1)")[0]
    assert isinstance(f, Finding)
    assert f.line == 1 and f.code == "LA01" and f.severity == "error"
    assert "LA01" in str(f)


def test_syntax_error_reported_as_la00(tmp_path):
    bad = tmp_path / "broken.py"
    bad.write_text("def oops(:\n    pass\n")
    findings = lint_file(str(bad))
    assert len(findings) == 1 and findings[0].code == "LA00"
