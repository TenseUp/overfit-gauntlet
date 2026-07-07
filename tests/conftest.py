"""Shared pytest fixtures and path setup for the overfit-gauntlet test suite.

Ensures the ``src`` layout is importable without an editable install, and
provides the two canonical, seeded return series used across the suite:

* ``edge_returns`` — a strong positive-edge series engineered to clear every bar
  of the gauntlet (verdict ``VALIDATED``).
* ``noise_returns`` — pure zero-mean noise that must NOT survive the gauntlet.

Both are deterministic (fixed ``numpy`` Generator seeds), so the verdict-level
assertions in ``test_gauntlet.py`` are reproducible on every run.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

# Per-period vol used to build the synthetic series (1% daily).
_VOL = 0.01
_N = 750


@pytest.fixture(scope="session")
def edge_returns() -> np.ndarray:
    """Seeded positive-edge series: annualized Sharpe ~2.5, n=750 -> VALIDATED."""
    rng = np.random.default_rng(0)
    mu = 2.5 * _VOL / np.sqrt(252)          # target annualized Sharpe ~2.5
    return rng.normal(mu, _VOL, _N)


@pytest.fixture(scope="session")
def noise_returns() -> np.ndarray:
    """Seeded zero-mean noise: no edge -> NOT VALIDATED."""
    rng = np.random.default_rng(1002)
    return rng.normal(0.0, _VOL, _N)


@pytest.fixture(scope="session")
def fixtures_dir() -> str:
    return FIXTURES
