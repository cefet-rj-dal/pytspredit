"""CEEMDAN prototype -- EVALUATED AND NOT ADOPTED in 0.7.0 (kept for reference).

Why it is here and not in ``src/``: the §1b "++" gate requires an improvement to
*predict/denoise as well or better* than the reference, not merely to be different.
On the decomposition filters' use case (``TsFilEmd``/``TsFilRemd`` remove the
high-frequency IMFs), CEEMDAN gave **no denoising gain over EEMD** while costing ~2.7x
more (measured on a synthetic noisy signal: CEEMDAN R2 0.9728 == EEMD R2 0.9728, EMD
0.9705). So it was left out to avoid gold-plating. This file preserves the validated
implementation in case a future use case (e.g. interpreting individual IMFs, where
CEEMDAN's reduced mode-mixing matters) justifies it.

Reference: Torres, M. E., Colominas, M. A., Schlotthauer, G., & Flandrin, P. (2011).
*A complete ensemble empirical mode decomposition with adaptive noise*. ICASSP.

Depends on the native ``pytspredit.emd.emd`` (single-pass EMD).
"""
from __future__ import annotations

import numpy as np

from pytspredit.emd import emd


def ceemdan(x, trials: int = 50, noise_width: float = 0.05, max_imf: int = 12,
            random_state=None) -> np.ndarray:
    """Complete Ensemble EMD with Adaptive Noise (Torres et al. 2011)."""
    x = np.asarray(x, float).ravel()
    n = len(x)
    rng = np.random.default_rng(random_state)
    white = [rng.standard_normal(n) for _ in range(trials)]
    white_imfs = [emd(w, max_imf=max_imf) for w in white]  # precompute noise modes

    eps0 = noise_width * np.std(x)
    first = np.mean([emd(x + eps0 * white[i], max_imf=1)[0] for i in range(trials)], axis=0)
    imfs = [first]
    residual = x - first
    k = 1
    while k < max_imf:
        d = np.diff(residual)
        if np.sum((np.r_[d, 0] < 0) & (np.r_[0, d] > 0)) < 2:
            break
        eps = noise_width * np.std(residual)
        acc = np.zeros(n)
        for i in range(trials):
            wk = white_imfs[i][k] if k < white_imfs[i].shape[0] else np.zeros(n)
            acc += emd(residual + eps * wk, max_imf=1)[0]
        d_k = acc / trials
        imfs.append(d_k)
        residual = residual - d_k
        k += 1
    imfs.append(residual)
    return np.asarray(imfs)
