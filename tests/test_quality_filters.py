"""R-vs-Python parity guard for the TsFil* family (retroactive M3 backfill, 0.4 theme).

Invariant = the full filtered series on the shared tsd with one injected outlier. Most
filters are deterministic and matched value-wise (<= 1e-9). ses is near-exact (~8e-5).
qes/seas_adj/spline use different backends (HoltWinters / stats::stl vs statsmodels STL /
smooth.spline vs scipy) and get a generous per-filter tolerance that catches a collapse
while surviving backend drift. EMD/wavelet filters have their own guards. Reference
develop/parity/filters_ref.json is produced by develop/parity/filters.R. See develop/parity.md.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

import pytspredit as ts

warnings.filterwarnings("ignore")

_REF = Path(__file__).resolve().parents[1] / "develop" / "parity" / "filters_ref.json"


def _load():
    with open(_REF, encoding="utf-8") as fh:
        return json.load(fh)


# filter name -> (builder, tolerance). Backend filters carry a loose, documented tol.
_FILTERS = {
    "none": (lambda: ts.TsFilNone(), 1e-9),
    "ma": (lambda: ts.TsFilMa(3), 1e-9),
    "ema": (lambda: ts.TsFilEma(3), 1e-9),
    "winsor": (lambda: ts.TsFilWinsor(), 1e-9),
    "recursive": (lambda: ts.TsFilRecursive(0.05), 1e-9),
    "fft": (lambda: ts.TsFilFft(), 1e-9),
    "smooth": (lambda: ts.TsFilSmooth(), 1e-9),
    "hp": (lambda: ts.TsFilHp(100), 1e-9),
    "gabor": (lambda: ts.TsFilGabor(3, 0.1), 1e-9),
    "lowess": (lambda: ts.TsFilLowess(0.2), 1e-9),
    "kalman": (lambda: ts.TsFilKalman(), 1e-6),
    "ses": (lambda: ts.TsFilSes(), 1e-3),
    "qes": (lambda: ts.TsFilQes(), 3e-1),        # HoltWinters backend
    "spline": (lambda: ts.TsFilSpline(0.5), 3e-1),  # smooth.spline vs scipy backend
    "seas_adj": (lambda: ts.TsFilSeasAdj(12), 2.0),  # stats::stl vs statsmodels STL backend
}


@pytest.mark.skipif(not _REF.exists(), reason="run develop/parity/filters.R")
@pytest.mark.parametrize("name", list(_FILTERS))
def test_filter_matches_r(name):
    ref = _load()
    y = np.asarray(ref["input"], float)
    builder, tol = _FILTERS[name]
    got = np.asarray(builder().fit(y).transform(y), float)
    expected = np.asarray(ref["filtered"][name], float)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, atol=tol, rtol=0, equal_nan=True,
                               err_msg=f"{name}: filtered series differs from R beyond {tol}")
