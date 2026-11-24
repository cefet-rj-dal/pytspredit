"""R-vs-Python parity guard for the TsLagmap family (retroactive M3 backfill).

Lag selection is a discrete decision; the invariant is the SET of selected lags, which
is convention-independent (R positions are 1-based, Python 0-based, so only the lag
numbers are compared -- exactly, no tolerance). Reference develop/parity/lagmap_ref.json
is produced by develop/parity/lagmap.R. See develop/parity.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import pytspredit as ts

_REF = Path(__file__).resolve().parents[1] / "develop" / "parity" / "lagmap_ref.json"


def _load():
    with open(_REF, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.mark.skipif(not _REF.exists(), reason="run develop/parity/lagmap.R")
@pytest.mark.parametrize("method", _load()["methods"] if _REF.exists() else [])
def test_lagmap_matches_r(method):
    ref = _load()
    io = ts.ts_projection(ts.ts_data(np.asarray(ref["input"], float), int(ref["sw"])))
    lm = ts.TsLagmap(method=method, seasonality=int(ref["seasonality"])).fit(
        io.input, io.output, input_size=int(ref["input_size"]))
    expected = sorted(int(v) for v in ref["lags"][method])
    got = sorted(int(v) for v in lm.lags)
    assert got == expected, f"{method}: lags {got} != R {expected}"
