"""``gauntlet`` console script.

Two subcommands:

* ``gauntlet run FILE`` reads a CSV of returns (and an optional cost column),
  runs the full gauntlet, prints the markdown report, and exits ``0`` only if the
  verdict is ``VALIDATED`` — so it doubles as a CI gate.
* ``gauntlet lint PATH...`` runs the static look-ahead linter over one or more
  files/directories, prints findings, and exits ``1`` if any error-severity
  finding is present.

Exit codes:
    0  success (VALIDATED, or lint clean of errors)
    1  gate failed (not VALIDATED, or lint found an error)
    2  usage / IO error
"""
from __future__ import annotations

import argparse
import csv
import sys
from typing import List, Optional, Sequence

from . import __version__
from .gauntlet import Verdict, run_gauntlet
from .lint import Finding, lint_path


# -- CSV reading -----------------------------------------------------------
def _read_csv_column(path: str, col: Optional[str]) -> tuple[List[float], List[str]]:
    """Read one numeric column from a CSV.

    If ``col`` is given, that named column is used; otherwise the single column
    (or the first column, if unambiguous) is used. Returns ``(values, header)``.
    """
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        rows = [row for row in reader if row]
    if not rows:
        raise ValueError(f"{path}: file is empty")

    header = rows[0]
    has_header = _looks_like_header(header)
    body = rows[1:] if has_header else rows

    idx = _resolve_column(path, header if has_header else None, col, len(header))
    values: List[float] = []
    for row in body:
        if idx >= len(row):
            continue
        cell = row[idx].strip()
        if cell == "":
            continue
        values.append(float(cell))
    return values, header


def _looks_like_header(row: Sequence[str]) -> bool:
    """A row is a header if at least one cell is non-numeric."""
    for cell in row:
        try:
            float(cell.strip())
        except ValueError:
            return True
    return False


def _resolve_column(path: str, header: Optional[Sequence[str]], col: Optional[str],
                    ncols: int) -> int:
    if col is not None:
        if header is None:
            # No header row — allow an integer index.
            try:
                return int(col)
            except ValueError:
                raise ValueError(f"{path}: no header, so --col must be an integer index")
        if col not in header:
            raise ValueError(f"{path}: column {col!r} not found; have {list(header)}")
        return header.index(col)
    # No column requested: use the first column.
    if ncols == 0:
        raise ValueError(f"{path}: no columns")
    return 0


# -- subcommands -----------------------------------------------------------
def _cmd_run(args: argparse.Namespace) -> int:
    try:
        returns, _ = _read_csv_column(args.file, args.col)
        costs = None
        if args.costs_col is not None:
            costs, _ = _read_csv_column(args.file, args.costs_col)
            if len(costs) != len(returns):
                print(
                    f"error: returns ({len(returns)}) and costs ({len(costs)}) "
                    "columns have different lengths",
                    file=sys.stderr,
                )
                return 2
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = run_gauntlet(
        returns,
        ann=args.ann,
        n_trials=args.trials,
        costs=costs,
    )
    print(report.to_markdown())
    return 0 if report.verdict is Verdict.VALIDATED else 1


def _cmd_lint(args: argparse.Namespace) -> int:
    findings: List[Finding] = []
    for path in args.paths:
        findings.extend(lint_path(path))
    findings.sort(key=lambda f: (f.file, f.line, f.code))

    if not findings:
        print("No findings. (A clean lint is necessary, never sufficient.)")
        return 0

    for f in findings:
        print(str(f))

    n_err = sum(1 for f in findings if f.severity == "error")
    n_warn = len(findings) - n_err
    print(f"\n{len(findings)} finding(s): {n_err} error, {n_warn} warn", file=sys.stderr)
    return 1 if n_err else 0


# -- entry point -----------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gauntlet",
        description="An anti-overfit validation gauntlet for trading backtests.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run the gauntlet on a returns CSV")
    run_p.add_argument("file", help="CSV file with a returns column")
    run_p.add_argument("--col", default=None, help="name (or index) of the returns column")
    run_p.add_argument("--ann", type=int, default=252, help="annualization factor (default 252)")
    run_p.add_argument("--trials", type=int, default=1,
                       help="number of strategies searched (deflated-Sharpe benchmark)")
    run_p.add_argument("--costs-col", default=None,
                       help="name (or index) of a per-period cost-drag column")
    run_p.set_defaults(func=_cmd_run)

    lint_p = sub.add_parser("lint", help="static look-ahead linter over .py files")
    lint_p.add_argument("paths", nargs="+", help="files or directories to scan")
    lint_p.set_defaults(func=_cmd_lint)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
