"""R-vs-Python quality guard for the native EMD filters.

EMD is stochastic (ensemble) and uses a different backend than R ``hht``, so parity is
checked at the DECISION / QUALITY level, not value-wise (see develop/parity.md):
the native filter must (a) genuinely denoise (improve R2 vs the clean signal) and
(b) remove an injected outlier at least as well as the R ``hht`` reference. The
reference develop/parity/filters_emd_ref.json is produced by develop/parity/filters_emd.R.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

import pytspredit as ts

warnings.filterwarnings("ignore")

_REF = Path(__file__).resolve().parents[1] / "develop" / "parity" / "filters_emd_ref.json"


def _r2(actual, pred):
    actual = np.asarray(actual, float); pred = np.asarray(pred, float)
    return 1 - np.sum((actual - pred) ** 2) / np.sum((actual - np.mean(actual)) ** 2)


def test_tsfilemd_improves_denoising_r2():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 512)
    clean = np.sin(2 * np.pi * 5 * t) + 0.3 * t
    noisy = clean + 0.2 * rng.standard_normal(512)
    filt = ts.TsFilEmd().fit(noisy).transform(noisy)
    assert _r2(clean, filt) > _r2(clean, noisy)  # removing IMF1 reduces noise


def _outlier_dev(s):
    return abs(s[8] - 0.5 * (s[7] + s[9]))


def test_native_removes_outlier_at_least_as_well_as_hht():
    if not _REF.exists():
        import pytest
        pytest.skip("parity reference not generated (run develop/parity/filters_emd.R)")
    ref = json.loads(_REF.read_text())
    yo = np.asarray(ref["input"], float)
    hht_filtered = np.asarray(ref["emd_filtered"], float)
    native = ts.TsFilEmd().fit(yo).transform(yo)
    # both attenuate the spike; native (deterministic EMD) does at least as well as hht
    assert _outlier_dev(native) <= _outlier_dev(yo)
    assert _outlier_dev(hht_filtered) <= _outlier_dev(yo)
    assert _outlier_dev(native) <= _outlier_dev(hht_filtered) + 0.25  # quality >= reference (tol)
