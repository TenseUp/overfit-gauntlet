"""Static lookahead linter for backtest code.

An AST-based scanner that walks ``.py`` files and flags the most common
sources of look-ahead bias and train/test leakage in backtesting and
model-validation code.

Honest scope: **this catches common sins, not all leakage.** It recognizes a
fixed set of syntactic patterns (rules ``LA01``–``LA08``). It cannot reason
about data flow, aliasing, or intent, so it will miss leakage expressed in
ways it does not pattern-match, and it can flag deliberate, correct uses. Treat
findings as prompts to think, not as proof of guilt or innocence. A clean lint
report is necessary, never sufficient.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from typing import List, Optional

__all__ = ["Finding", "lint_source", "lint_file", "lint_path"]


@dataclass(frozen=True)
class Finding:
    """A single lint result.

    Attributes:
        file: Path to the source file (``"<string>"`` for in-memory sources).
        line: 1-based line number of the offending node.
        code: Rule code, one of ``LA01``..``LA08``.
        message: Human-readable explanation of the risk.
        severity: ``"error"`` for near-certain leakage, ``"warn"`` for heuristics.
    """

    file: str
    line: int
    code: str
    message: str
    severity: str = "error"

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        return f"{self.file}:{self.line}: {self.code} [{self.severity}] {self.message}"


def _attr_name(node: ast.AST) -> Optional[str]:
    """Return the attribute name of a call target, e.g. ``df.shift`` -> ``shift``."""
    func = getattr(node, "func", None)
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _func_name(node: ast.AST) -> Optional[str]:
    """Return the bare name of a called function, e.g. ``KFold(...)`` -> ``KFold``."""
    func = getattr(node, "func", None)
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _first_positional_negative(call: ast.Call) -> bool:
    """True if the first positional arg is a negative numeric literal."""
    if not call.args:
        return False
    return _is_negative_number(call.args[0])


def _is_negative_number(node: ast.AST) -> bool:
    """True for a negative numeric literal such as ``-1`` or ``-3``."""
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
        and not isinstance(node.operand.value, bool)
    ):
        return node.operand.value != 0
    # Some parsers fold to a plain negative Constant.
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return node.value < 0
    return False


def _keyword(call: ast.Call, name: str) -> Optional[ast.keyword]:
    for kw in call.keywords:
        if kw.arg == name:
            return kw
    return None


def _is_true(node: Optional[ast.AST]) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _is_false(node: Optional[ast.AST]) -> bool:
    return isinstance(node, ast.Constant) and node.value is False


def _str_value(node: Optional[ast.AST]) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


# Method-name hints for LA06 (fit / fit_transform of a transformer before split).
_TRANSFORMER_HINTS = (
    "scaler",
    "scale",
    "standardscaler",
    "minmaxscaler",
    "robustscaler",
    "normalizer",
    "pca",
    "transformer",
    "encoder",
    "imputer",
    "selector",
)

_SPLIT_CALLS = ("train_test_split",)


class _Linter(ast.NodeVisitor):
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.findings: List[Finding] = []
        self._fit_calls: List[ast.Call] = []
        self._split_lines: List[int] = []

    def _add(self, node: ast.AST, code: str, message: str, severity: str = "error") -> None:
        self.findings.append(
            Finding(
                file=self.filename,
                line=getattr(node, "lineno", 0),
                code=code,
                message=message,
                severity=severity,
            )
        )

    # -- LA07: full-sample normalization: (x - x.mean()) / x.std() ---------
    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Div):
            denom = node.right
            numer = node.left
            if (
                self._is_whole_series_method(denom, "std")
                and isinstance(numer, ast.BinOp)
                and isinstance(numer.op, ast.Sub)
                and self._is_whole_series_method(numer.right, "mean")
            ):
                self._add(
                    node,
                    "LA07",
                    "full-sample normalization: mean/std computed over the whole "
                    "series leaks test-set statistics into training; fit scaling on "
                    "train only",
                    severity="warn",
                )
        self.generic_visit(node)

    @staticmethod
    def _is_whole_series_method(node: ast.AST, method: str) -> bool:
        """True for a no-arg call like ``x.mean()`` (whole-series reduction)."""
        return (
            isinstance(node, ast.Call)
            and not node.args
            and not node.keywords
            and _attr_name(node) == method
        )

    def visit_Call(self, node: ast.Call) -> None:
        attr = _attr_name(node)
        fname = _func_name(node)

        # -- LA01: .shift(-n) future data ---------------------------------
        if attr == "shift" and _first_positional_negative(node):
            self._add(
                node,
                "LA01",
                "negative .shift() pulls future rows into the present (look-ahead)",
            )
        neg_kw = _keyword(node, "periods")
        if attr == "shift" and neg_kw is not None and _is_negative_number(neg_kw.value):
            self._add(
                node,
                "LA01",
                "negative .shift(periods=...) pulls future rows into the present",
            )

        # -- LA08: .pct_change(-n) future data ----------------------------
        if attr == "pct_change" and _first_positional_negative(node):
            self._add(
                node,
                "LA08",
                "negative .pct_change() computes returns against future values",
            )
        pc_kw = _keyword(node, "periods")
        if attr == "pct_change" and pc_kw is not None and _is_negative_number(pc_kw.value):
            self._add(
                node,
                "LA08",
                "negative .pct_change(periods=...) computes returns against the future",
            )

        # -- LA02: .rolling(center=True) sees the future ------------------
        if attr == "rolling":
            center = _keyword(node, "center")
            if center is not None and _is_true(center.value):
                self._add(
                    node,
                    "LA02",
                    "rolling(center=True) centers the window on the current row, so "
                    "it includes future observations",
                )

        # -- LA03: backfill leaks -----------------------------------------
        if attr == "bfill" or attr == "backfill":
            self._add(
                node,
                "LA03",
                ".bfill()/.backfill() propagates future values backward in time",
            )
        if attr == "fillna":
            method = _keyword(node, "method")
            if method is not None and _str_value(method.value) in ("bfill", "backfill"):
                self._add(
                    node,
                    "LA03",
                    "fillna(method='bfill') propagates future values backward in time",
                )

        # -- LA04: train_test_split without shuffle=False -----------------
        if fname == "train_test_split":
            self._split_lines.append(getattr(node, "lineno", 0))
            shuffle = _keyword(node, "shuffle")
            if shuffle is None or not _is_false(shuffle.value):
                self._add(
                    node,
                    "LA04",
                    "train_test_split() without shuffle=False shuffles rows, "
                    "destroying temporal order of a time series",
                )

        # -- LA05: KFold(shuffle=True) / non-TimeSeriesSplit CV -----------
        if fname in ("KFold", "StratifiedKFold", "GroupKFold", "RepeatedKFold"):
            shuffle = _keyword(node, "shuffle")
            if shuffle is not None and _is_true(shuffle.value):
                self._add(
                    node,
                    "LA05",
                    f"{fname}(shuffle=True) breaks temporal ordering; use "
                    "TimeSeriesSplit or purged CV for time series",
                )
            else:
                self._add(
                    node,
                    "LA05",
                    f"{fname} performs plain k-fold CV; on time series it trains on "
                    "future folds — use TimeSeriesSplit or purged CV",
                    severity="warn",
                )

        # -- LA06: record fit / fit_transform of a transformer ------------
        if attr in ("fit", "fit_transform"):
            self._fit_calls.append(node)

        self.generic_visit(node)

    def finalize(self) -> None:
        """Emit LA06 findings for transformers fit before the first split."""
        if not self._split_lines:
            return
        first_split = min(self._split_lines)
        for call in self._fit_calls:
            line = getattr(call, "lineno", 0)
            if line >= first_split:
                continue
            if not self._looks_like_transformer(call):
                continue
            self._add(
                call,
                "LA06",
                "scaler/transformer fit before the train/test split leaks whole-"
                "dataset statistics; fit on the training split only",
            )

    @staticmethod
    def _looks_like_transformer(call: ast.Call) -> bool:
        """Heuristic: does the receiver of ``.fit``/``.fit_transform`` look like a
        scaler/PCA/transformer rather than a model?"""
        func = call.func
        if not isinstance(func, ast.Attribute):
            return False
        receiver = func.value
        name = None
        if isinstance(receiver, ast.Name):
            name = receiver.id
        elif isinstance(receiver, ast.Call):
            name = _func_name(receiver)
        elif isinstance(receiver, ast.Attribute):
            name = receiver.attr
        if name is None:
            return False
        low = name.lower()
        return any(hint in low for hint in _TRANSFORMER_HINTS)


def lint_source(source: str, filename: str = "<string>") -> List[Finding]:
    """Lint a source string and return findings sorted by (line, code)."""
    tree = ast.parse(source, filename=filename)
    linter = _Linter(filename)
    linter.visit(tree)
    linter.finalize()
    return sorted(linter.findings, key=lambda f: (f.line, f.code))


def lint_file(path: str) -> List[Finding]:
    """Lint a single ``.py`` file. Syntax errors are reported as an ``LA00`` finding."""
    with open(path, "r", encoding="utf-8") as fh:
        source = fh.read()
    try:
        return lint_source(source, filename=path)
    except SyntaxError as exc:
        return [
            Finding(
                file=path,
                line=exc.lineno or 0,
                code="LA00",
                message=f"could not parse file: {exc.msg}",
                severity="warn",
            )
        ]


def _iter_py_files(path: str):
    if os.path.isfile(path):
        if path.endswith(".py"):
            yield path
        return
    for root, dirs, files in os.walk(path):
        # Skip common noise directories.
        dirs[:] = [
            d
            for d in dirs
            if d not in {".git", "__pycache__", ".venv", "venv", ".tox", "build", "dist"}
        ]
        for name in sorted(files):
            if name.endswith(".py"):
                yield os.path.join(root, name)


def lint_path(path: str) -> List[Finding]:
    """Collect and lint every ``.py`` file under ``path``.

    ``path`` may be a single file or a directory (walked recursively). Findings
    are returned sorted by ``(file, line, code)``.
    """
    findings: List[Finding] = []
    for py in _iter_py_files(path):
        findings.extend(lint_file(py))
    return sorted(findings, key=lambda f: (f.file, f.line, f.code))
