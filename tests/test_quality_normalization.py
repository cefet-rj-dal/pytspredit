"""R-vs-Python parity guard for the TsNorm* family (retroactive M3 backfill).

Normalizations are deterministic transforms, so the full transformed window matrix is
checked VALUE-WISE against the R ``tspredit`` reference (tol <= 1e-9). Reference
develop/parity/normalization_ref.json is produced by develop/parity/normalization.R.
See develop/parity.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import pytspredit as ts

_REF = Path(__file__).resolve().parents[1] / "develop" / "parity" / "normalization_ref.json"


def _load():
    with open(_REF, encoding="utf-8") as fh:
        return json.load(fh)


# config name -> Python TsNorm* builder (mirrors develop/parity/normalization.R)
_BUILDERS = {
    "none": lambda: ts.TsNormNone(),
    "gminmax": lambda: ts.TsNormGminmax(),
    "swminmax": lambda: ts.TsNormSwminmax(),
    "diff": lambda: ts.TsNormDiff(),
    "an_default": lambda: ts.TsNormAn(),
    "an_subtract": lambda: ts.TsNormAn(operation="subtract"),
    "an_ema_softdiv_mad": lambda: ts.TsNormAn(average="ema", operation="softdivide", scale="mad"),
    "an_asinh_sd": lambda: ts.TsNormAn(operation="asinh", scale="sd"),
}


@pytest.mark.skipif(not _REF.exists(), reason="run develop/parity/normalization.R")
@pytest.mark.parametrize("name", _load()["configs"] if _REF.exists() else [])
def test_normalization_matches_r(name):
    ref = _load()
    w = ts.ts_data(np.asarray(ref["input"], float), int(ref["sw"]))
    expected = np.asarray(ref["transformed"][name], float)
    got = np.asarray(_BUILDERS[name]().fit(w).transform(w), float)
    assert got.shape == expected.shape, f"{name}: shape {got.shape} != R {expected.shape}"
    np.testing.assert_allclose(got, expected, atol=1e-9, rtol=0,
                               err_msg=f"{name}: transformed matrix differs from R")
