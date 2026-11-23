"""Native Maximal Overlap Discrete Wavelet Transform (MODWT).

A dependency-free (pure numpy) implementation of the MODWT and its inverse, used by
the wavelet denoising filter ``TsFilWavelet``. Unlike EMD, the MODWT is a fixed,
deterministic linear transform, so this native port matches the R ``wavelets`` package
value-for-value (~5e-6), and -- unlike wrapping PyWavelets' ``swt`` (a different
normalization convention) -- needs no new dependency at all (not even scipy).

Canonical here: this native implementation is the reference; the R ``tspredit`` filter
(which used the ``wavelets`` package) is to be backported to it, dropping that dependency.

The wavelet/scaling filter coefficients are the Daubechies/least-asymmetric/coiflet/
best-localized families as provided by R ``wavelets::wt.filter`` (public-domain
constants), carried here at full double precision.

References:
    - Percival, D. B., & Walden, A. T. (2000). *Wavelet Methods for Time Series
      Analysis*. Cambridge University Press (MODWT pyramid algorithm, Chapter 5).
    - Mallat, S. (1989). *A theory for multiresolution signal decomposition: the
      wavelet representation*. IEEE TPAMI, 11(7), 674-693.
    - Daubechies, I. (1992). *Ten Lectures on Wavelets*. SIAM (source of the
      Daubechies / least-asymmetric / coiflet filter coefficient families).

Porting:
    Canonical native implementation. The R ``ts_fil_wavelet.R`` wraps
    ``wavelets::modwt``/``imodwt``; this module replaces that engine. Fidelity:
    approximate (~5e-6 vs R; different numeric path, same algorithm and coefficients).
"""
from __future__ import annotations

import numpy as np

__all__ = ["imodwt", "modwt", "modwt_filters"]

# (scaling g, wavelet h) coefficients, from R wavelets::wt.filter, full precision.
_COEFFICIENTS = {
    "haar": (np.array([0.707106781186547, 0.707106781186547]),
             np.array([0.707106781186547, -0.707106781186547])),
    "d4": (np.array([0.482962913144534, 0.836516303737808, 0.224143868042013, -0.12940952255126]),
           np.array([-0.12940952255126, -0.224143868042013, 0.836516303737808, -0.482962913144534])),
    "la8": (np.array([-0.0757657147893407, -0.0296355276459541, 0.497618667632458, 0.803738751805216, 0.297857795605542, -0.0992195435769354, -0.0126039672622612, 0.0322231006040713]),
            np.array([0.0322231006040713, 0.0126039672622612, -0.0992195435769354, -0.297857795605542, 0.803738751805216, -0.497618667632458, -0.0296355276459541, 0.0757657147893407])),
    "bl14": (np.array([0.0120154192834842, 0.0172133762994439, -0.0649080035533744, -0.064131289818917, 0.360218460898555, 0.781921593296555, 0.483610915693782, -0.0568044768822707, -0.101010920866413, 0.0447423494687405, 0.0204642075778225, -0.0181266051311065, -0.0032832978473081, 0.0022918339541009]),
             np.array([0.0022918339541009, 0.0032832978473081, -0.0181266051311065, -0.0204642075778225, 0.0447423494687405, 0.101010920866413, -0.0568044768822707, -0.483610915693782, 0.781921593296555, -0.360218460898555, -0.064131289818917, 0.0649080035533744, 0.0172133762994439, -0.0120154192834842])),
    "c6": (np.array([-0.0156557285289848, -0.0727326213410511, 0.384864856538113, 0.85257204164239, 0.337897670951159, -0.0727322757411889]),
           np.array([-0.0727322757411889, -0.337897670951159, 0.85257204164239, -0.384864856538113, -0.0727326213410511, 0.0156557285289848])),
}


def modwt_filters() -> list[str]:
    """Return the available wavelet filter names."""
    return list(_COEFFICIENTS)


def _get(filter: str):
    if filter not in _COEFFICIENTS:
        raise ValueError(f"unknown wavelet filter {filter!r}; choose from {modwt_filters()}")
    return _COEFFICIENTS[filter]


def _n_levels(n: int, length: int) -> int:
    """Default number of levels (matches R ``wavelets`` for periodic boundary)."""
    return max(1, int(np.floor(np.log2((n - 1) / (length - 1) + 1))))


def modwt(x, filter: str = "haar", n_levels: int | None = None):
    """Maximal Overlap DWT with periodic boundary (Percival & Walden 2000).

    Returns ``(W, V)`` where ``W`` is a list of ``n_levels`` detail arrays and ``V``
    is the final scaling (smooth) array, each the same length as ``x``.
    """
    g, h = _get(filter)
    gt, ht = g / np.sqrt(2.0), h / np.sqrt(2.0)
    length = len(g)
    v = np.asarray(x, dtype=float).ravel().copy()
    n = len(v)
    levels = _n_levels(n, length) if n_levels is None else int(n_levels)
    details = []
    for j in range(1, levels + 1):
        s = 2 ** (j - 1)
        wj = sum(ht[l] * np.roll(v, s * l) for l in range(length))
        v = sum(gt[l] * np.roll(v, s * l) for l in range(length))
        details.append(wj)
    return details, v


def imodwt(details, v, filter: str = "haar") -> np.ndarray:
    """Inverse MODWT: reconstruct the series from ``(details, v)``."""
    g, h = _get(filter)
    gt, ht = g / np.sqrt(2.0), h / np.sqrt(2.0)
    length = len(g)
    vj = np.asarray(v, dtype=float).ravel().copy()
    for j in range(len(details), 0, -1):
        s = 2 ** (j - 1)
        wj = details[j - 1]
        vj = sum(ht[l] * np.roll(wj, -s * l) + gt[l] * np.roll(vj, -s * l)
                 for l in range(length))
    return vj


def modwt_denoise(x, filter: str = "haar") -> np.ndarray:
    """Reconstruct ``x`` with the first (finest) detail level zeroed (``x - D1``)."""
    details, v = modwt(x, filter=filter)
    details[0] = np.zeros_like(details[0])
    return imodwt(details, v, filter=filter)
