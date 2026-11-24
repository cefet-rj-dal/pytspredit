"""Empirical Mode Decomposition (EMD / EEMD).

Implementation of EMD and its noise-assisted ensemble variant (EEMD), used by the
decomposition filters ``TsFilEmd`` and ``TsFilRemd``. Decomposes a signal into Intrinsic
Mode Functions (IMFs) by iterative sifting with cubic-spline envelopes and Rilling-style
boundary mirroring.

Theory:
    EMD is a data-driven, fully adaptive decomposition for nonlinear and non-stationary
    signals (Huang et al. 1998). It extracts IMFs -- zero-mean oscillatory components with
    a locally symmetric envelope -- by repeatedly *sifting*: interpolate the upper and
    lower extrema with cubic-spline envelopes, subtract their mean, and iterate until the
    component is an IMF; the IMF is then removed and the process repeats on the residual.
    EEMD (Wu & Huang 2009) adds controlled white noise across an ensemble and averages the
    resulting IMFs to mitigate mode mixing. End effects are handled by reflecting the
    outermost extrema across the boundaries (Rilling et al. 2003).

Canonical attribution:
    This module is the canonical reference implementation of EMD/EEMD for the toolkit, and
    the R ``tspredit`` decomposition filters are backported to the same algorithm. The
    improvement (++), with evidence: exact reconstruction (the IMFs sum to the signal to
    ~4e-16), denoising R-squared that matches or exceeds reference implementations on the
    package's outlier-removal use case, and roughly three orders of magnitude faster than
    the ``hht``-based reference at n~2048. See ``develop/parity.md`` Section 5.4.

References:
    - Huang, N. E., Shen, Z., Long, S. R., et al. (1998). *The empirical mode
      decomposition and the Hilbert spectrum for nonlinear and non-stationary time
      series analysis*. Proceedings of the Royal Society A, 454, 903-995.
    - Wu, Z., & Huang, N. E. (2009). *Ensemble Empirical Mode Decomposition: a
      noise-assisted data analysis method*. Advances in Adaptive Data Analysis, 1(1), 1-41.
    - Rilling, G., Flandrin, P., & Goncalves, P. (2003). *On empirical mode
      decomposition and its algorithms*. IEEE-EURASIP NSIP (boundary mirroring).

Porting:
    Mirrors the R native decomposition core (``emd_decompose`` / ``eemd_decompose`` in
    ``ts_fil_emd.R``), which this module is the reference for. Fidelity: adapted -- EMD is
    adaptive and (for EEMD) stochastic, so parity is quality-level (the denoising outcome),
    not value-wise.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

__all__ = ["eemd", "emd"]


def _extrema(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Indices of interior local maxima and minima (via derivative sign change)."""
    d = np.diff(x)
    maxima = np.where((np.r_[d, 0] < 0) & (np.r_[0, d] > 0))[0]
    minima = np.where((np.r_[d, 0] > 0) & (np.r_[0, d] < 0))[0]
    return maxima, minima


def _mirror(idx: np.ndarray, vals: np.ndarray, n: int, k: int = 2):
    """Reflect the first/last ``k`` extrema across the boundaries (Rilling 2003)."""
    left_i = -idx[:k][::-1]
    left_v = vals[:k][::-1]
    right_i = 2 * (n - 1) - idx[-k:][::-1]
    right_v = vals[-k:][::-1]
    return np.r_[left_i, idx, right_i], np.r_[left_v, vals, right_v]


def _envelope(idx: np.ndarray, x: np.ndarray, n: int) -> np.ndarray:
    ei, ev = _mirror(idx, x[idx], n)
    return CubicSpline(ei, ev)(np.arange(n))


def emd(x, max_imf: int = 12, sift_max: int = 100, sd_tol: float = 0.2) -> np.ndarray:
    """Empirical Mode Decomposition of a 1-D signal.

    Returns an array of shape ``(n_imf, len(x))`` whose rows sum exactly to ``x`` (the
    last row is the residual trend). Sifting stops when the normalized squared
    difference between successive sift iterates falls below ``sd_tol`` (Huang 1998),
    or after ``sift_max`` iterations; IMF extraction stops when the residual has too
    few extrema.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    imfs: list[np.ndarray] = []
    residual = x.copy()
    for _ in range(max_imf):
        h = residual.copy()
        for _ in range(sift_max):
            maxima, minima = _extrema(h)
            if len(maxima) < 2 or len(minima) < 2:
                break
            mean = 0.5 * (_envelope(maxima, h, n) + _envelope(minima, h, n))
            h_next = h - mean
            sd = np.sum((h - h_next) ** 2) / (np.sum(h ** 2) + 1e-12)
            h = h_next
            if sd < sd_tol:
                break
        imfs.append(h)
        residual = residual - h
        maxima, minima = _extrema(residual)
        if len(maxima) + len(minima) < 2:
            break
    imfs.append(residual)
    return np.asarray(imfs)


def eemd(x, trials: int = 100, noise_width: float = 0.05, max_imf: int = 12,
         random_state=None) -> np.ndarray:
    """Ensemble EMD: average IMFs over ``trials`` noisy copies (Wu & Huang 2009).

    White noise of standard deviation ``noise_width * std(x)`` is added to each copy;
    IMFs are averaged position-wise. ``random_state`` seeds the noise for
    reproducibility (``None`` -> non-deterministic, as in the R ensemble).
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    rng = np.random.default_rng(random_state)
    scale = noise_width * np.std(x)
    acc = np.zeros((max_imf + 2, n))
    counts = np.zeros(max_imf + 2)
    for _ in range(trials):
        im = emd(x + rng.standard_normal(n) * scale, max_imf=max_imf)
        k = im.shape[0]
        acc[:k] += im
        counts[:k] += 1
    k = int(np.sum(counts > 0))
    return acc[:k] / np.maximum(counts[:k, None], 1.0)
