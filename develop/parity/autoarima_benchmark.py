"""Benchmark: ARIMA++ (auto="cv") vs the compact AICc auto on real datasets.

This is the §1b gate for 0.10.0: the ACF/PACF-driven, horizon-matched cross-validated
selector must forecast at least as well as the AICc baseline (which approximates
forecast::auto.arima). Run manually (slow -- CV fits many ARIMA models):

    python develop/parity/autoarima_benchmark.py

It pulls real competition series via datasets.loadfulldata, holds out the last H points,
and reports mean sMAPE. (Optionally compare against R forecast::auto.arima by exporting
the same series and running it in R; the AICc mode here is the in-package proxy.)

Recorded result (NN5, first 4 series, n<=200, H=14, 2026-11-23):
    MEAN sMAPE  aicc = 0.1565   cv++ = 0.1352   (cv++ ~14% better; wins 4/4)
    Orders: cv++ recovered the weekly lag (e.g. (7,0,7), (7,0,2)) that the AICc cap of
    5 misses. Key design point: the CV horizon must MATCH the forecast horizon --
    selecting by 1-step CV instead gave cv++ 0.1655 (worse); horizon-matched gave 0.1352.
"""
from __future__ import annotations

import warnings

import numpy as np

import pytspredit as ts

warnings.filterwarnings("ignore")


def smape(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    return float(np.mean(2 * np.abs(a - p) / (np.abs(a) + np.abs(p) + 1e-9)))


def run(dataset="NN5", n_series=4, max_len=200, horizon=14):
    data = ts.datasets.loadfulldata(dataset)
    cols = [c for c in data.columns if c.lower() != "split"][:n_series]
    aicc, cvpp = [], []
    for c in cols:
        s = data[c].dropna().to_numpy()[:max_len]
        tr, te = s[:-horizon], s[-horizon:]
        pa = ts.TsArima(auto="aicc").fit(tr).predict(te, steps_ahead=horizon)
        mc = ts.TsArima(auto="cv", horizon=horizon).fit(tr)
        pc = mc.predict(te, steps_ahead=horizon)
        aicc.append(smape(te, pa))
        cvpp.append(smape(te, pc))
        print(f"{c}: aicc sMAPE={aicc[-1]:.4f} | cv++ {mc._order()} sMAPE={cvpp[-1]:.4f}")
    print(f"MEAN sMAPE  aicc={np.mean(aicc):.4f}  cv++={np.mean(cvpp):.4f}")


if __name__ == "__main__":
    run()
