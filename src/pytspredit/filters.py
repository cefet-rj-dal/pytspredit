"""Time series filters (smoothing / denoising).

Python counterpart of the R ``ts_fil_*`` files. A filter is a
:class:`pydaltoolbox.DalTransform` that takes a 1-D series and returns a filtered
1-D series of the same length.

Parity with R:

* **Exact (native / manual ports):** ``none``, ``ma``, ``ema``, ``winsor``,
  ``recursive``, ``fft``, ``smooth``, ``hp`` (manual linear solve) and ``gabor``
  (manual kernel + convolution).
* **Backend-dependent (documented tolerance):** ``ses``/``qes`` (R
  ``HoltWinters`` vs statsmodels), ``lowess``, ``seas_adj`` (STL), ``spline`` (R
  ``smooth.spline`` vs scipy) and ``kalman`` (R ``KFAS`` vs a local-level Kalman
  filter). Same spirit as the ARIMA/RF/MLP tolerances.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pydaltoolbox import DalTransform

from .augment import _boxplot_outlier_idx
from .emd import eemd, emd
from .tsdata import ts_data
from .wavelet import modwt_denoise

__all__ = [
    "TsFilEma", "TsFilEmd", "TsFilFft", "TsFilGabor", "TsFilHp", "TsFilKalman",
    "TsFilLowess", "TsFilMa", "TsFilNone", "TsFilQes", "TsFilRecursive", "TsFilRemd",
    "TsFilSeasAdj", "TsFilSes", "TsFilSmooth", "TsFilSpline", "TsFilWavelet",
    "TsFilWinsor",
]


def _vec(data) -> np.ndarray:
    if isinstance(data, (pd.Series, pd.DataFrame)):
        return data.to_numpy(dtype=float).ravel()
    return np.asarray(data, dtype=float).ravel()


class _Filter(DalTransform):
    def fit(self, data=None, **kwargs) -> _Filter:
        return self


class TsFilNone(_Filter):
    """Identity filter (``ts_fil_none``).

    Porting:
        Mirrors ``ts_fil_none.R``. Fidelity: exact.
    """

    def transform(self, data, **kwargs):
        return data


class TsFilMa(_Filter):
    """Moving average (``ts_fil_ma``).

    Porting:
        Mirrors ``ts_fil_ma.R``. Fidelity: exact (native, ~5e-9).
    """

    def __init__(self, ma: int = 3) -> None:
        self.ma = ma

    def transform(self, data, **kwargs) -> np.ndarray:
        w = ts_data(_vec(data), self.ma).to_numpy(dtype=float)
        return np.concatenate([np.full(self.ma - 1, np.nan), w.mean(axis=1)])


class TsFilEma(_Filter):
    """Exponential moving average (``ts_fil_ema``).

    References:
        - Holt CC (1957). Forecasting trends and seasonals by exponentially
          weighted moving averages. O.N.R. Research Memorandum.

    Porting:
        Mirrors ``ts_fil_ema.R``. Fidelity: exact (native, ~5e-9).
    """

    def __init__(self, ema: int = 3) -> None:
        self.ema = ema

    def transform(self, data, **kwargs) -> np.ndarray:
        w = ts_data(_vec(data), self.ema).to_numpy(dtype=float)
        n = self.ema
        alfa = 1 - 2.0 / (n + 1)
        weights = alfa ** np.arange(n - 1, -1, -1)  # newest -> 1
        ema = (w * weights).sum(axis=1) / weights.sum()
        return np.concatenate([np.full(self.ema - 1, np.nan), ema])


class TsFilWinsor(_Filter):
    """Winsorization to the 5th/95th percentiles (``ts_fil_winsor``).

    References:
        - Tukey JW (1962). The future of data analysis. Annals of Mathematical
          Statistics, 33(1), 1-67. (Winsorization in robust summaries)

    Porting:
        Mirrors ``ts_fil_winsor.R``. Fidelity: exact (native, ~5e-9).
    """

    def transform(self, data, **kwargs) -> np.ndarray:
        x = _vec(data)
        lo, hi = np.quantile(x, [0.05, 0.95])  # numpy default == R quantile type 7
        return np.clip(x, lo, hi)


class TsFilRecursive(_Filter):
    """Recursive linear filter ``y[i] = x[i] + f*y[i-1]`` (``ts_fil_recursive``).

    Porting:
        Mirrors ``ts_fil_recursive.R``. Fidelity: exact (native, ~5e-9).
    """

    def __init__(self, filter: float) -> None:
        self.filter = filter

    def transform(self, data, **kwargs) -> np.ndarray:
        x = _vec(data)
        y = np.empty_like(x)
        prev = 0.0
        for i in range(len(x)):
            prev = x[i] + self.filter * prev
            y[i] = prev
        return y


class TsFilSmooth(_Filter):
    """Robust progressive smoothing of outlier jumps (``ts_fil_smooth``).

    Porting:
        Mirrors ``ts_fil_smooth.R``. Fidelity: exact (native, ~5e-9).
    """

    def transform(self, data, **kwargs) -> np.ndarray:
        s = _vec(data)
        s = s[np.isfinite(s)]
        while True:
            diff = s[1:] - s[:-1]
            out = _boxplot_outlier_idx(diff)
            rj = out[(out > 1) & (out < len(s))]
            s[rj - 1] = (s[rj - 2] + s[rj]) / 2
            out2 = _boxplot_outlier_idx(s[1:] - s[:-1])
            if len(out2) == 0 or len(out2) == len(out):
                break
        return s


class TsFilHp(_Filter):
    """Hodrick-Prescott filter, blended with the original (``ts_fil_hp``).

    References:
        - Hodrick RJ, Prescott EC (1997). Postwar U.S. business cycles: an
          empirical investigation. Journal of Money, Credit and Banking, 29(1).

    Porting:
        Mirrors ``ts_fil_hp.R``. Fidelity: exact (manual HP linear solve, ~5e-9;
        not statsmodels ``hpfilter``, for bit-level parity).
    """

    def __init__(self, lambda_: float = 100.0, preserve: float = 0.9) -> None:
        if not 0 <= preserve <= 1:
            raise ValueError("preserve must be in [0, 1].")
        self.lambda_ = lambda_
        self.preserve = preserve

    def _trend(self, y: np.ndarray) -> np.ndarray:
        n = len(y)
        if n < 3:
            return y
        d2 = np.diff(np.eye(n), n=2, axis=0)      # second-difference matrix
        penalty = self.lambda_ * (d2.T @ d2)
        return np.linalg.solve(np.eye(n) + penalty, y)

    def transform(self, data, **kwargs) -> np.ndarray:
        y = _vec(data)
        trend = self._trend(y)
        return self.preserve * y + (1 - self.preserve) * trend


class TsFilFft(_Filter):
    """FFT low-pass smoothing with reflected padding + cosine roll-off (``ts_fil_fft``).

    References:
        - Cooley JW, Tukey JW (1965). An algorithm for the machine calculation of
          complex Fourier series. Mathematics of Computation, 19, 297-301.

    Porting:
        Mirrors ``ts_fil_fft.R``. Fidelity: exact (native, ~5e-9).
    """

    def __init__(self, threshold: float = 0.90, pad_pct: float = 0.25, taper_pct: float = 0.15) -> None:
        if not (0 < threshold <= 1):
            raise ValueError("Invalid threshold value")
        if pad_pct < 0:
            raise ValueError("Invalid pad_pct value")
        if taper_pct < 0:
            raise ValueError("Invalid taper_pct value")
        self.threshold = threshold
        self.pad_pct = pad_pct
        self.taper_pct = taper_pct
        self.cutindex = None
        self.pad_len = None

    def _pad_length(self, n: int) -> int:
        return min(max(1, int(np.floor(n * self.pad_pct))), n - 1)

    @staticmethod
    def _reflect_pad(x: np.ndarray, pad_len: int) -> np.ndarray:
        n = len(x)
        left = 2 * x[0] - x[pad_len:0:-1]            # x[pad_len], ..., x[1]
        right = 2 * x[n - 1] - x[n - 2:n - pad_len - 2:-1]  # x[n-2], ..., x[n-pad_len-1]
        return np.concatenate([left, x, right])

    def _cut_index(self, spectrum_half: np.ndarray) -> int:
        if len(spectrum_half) <= 1:
            return 1
        ac = spectrum_half[1:]
        total = ac.sum()
        if total <= np.finfo(float).eps:
            return 1
        cum = np.cumsum(ac) / total
        idx = np.argmax(cum >= self.threshold)
        cut_ac = idx + 1 if cum[idx] >= self.threshold else len(ac)
        return int(cut_ac + 1)

    @staticmethod
    def _lowpass_weights(n: int, half_len: int, cutindex: int, taper_pct: float) -> np.ndarray:
        w = np.zeros(half_len)
        taper = max(1, int(np.floor(cutindex * taper_pct)))
        for k in range(1, half_len + 1):
            if k <= cutindex:
                w[k - 1] = 1.0
            elif k <= cutindex + taper:
                w[k - 1] = 0.5 * (1 + np.cos(np.pi * (k - cutindex) / taper))
        full = np.zeros(n)
        full[:half_len] = w
        if n > 2:
            full[half_len:n] = w[1:n - half_len + 1][::-1]
        return full

    def fit(self, data, **kwargs) -> TsFilFft:
        x = _vec(data)
        n = len(x)
        if n < 4 or np.any(np.isnan(x)):
            return self
        pad_len = self._pad_length(n)
        padded = self._reflect_pad(x, pad_len)
        spec = np.abs(np.fft.fft(padded)) ** 2
        half = int(np.floor(len(padded) / 2) + 1)
        self.cutindex = self._cut_index(spec[:half])
        self.pad_len = pad_len
        return self

    def transform(self, data, **kwargs) -> np.ndarray:
        x = _vec(data)
        n = len(x)
        if n < 4 or np.any(np.isnan(x)):
            return x
        pad_len = self._pad_length(n) if self.pad_len is None else self.pad_len
        padded = self._reflect_pad(x, pad_len)
        n_pad = len(padded)
        sig = np.fft.fft(padded)
        half = int(np.floor(n_pad / 2) + 1)
        cutindex = self.cutindex
        if cutindex is None:
            cutindex = self._cut_index((np.abs(sig) ** 2)[:half])
        cutindex = min(max(1, cutindex), half)
        weights = self._lowpass_weights(n_pad, half, cutindex, self.taper_pct)
        recon = np.real(np.fft.ifft(sig * weights))
        return recon[pad_len:pad_len + n]


class TsFilGabor(_Filter):
    """One-dimensional Gabor smoothing (``ts_fil_gabor``).

    References:
        - Gabor D (1946). Theory of communication. Journal of the IEE - Part III,
          93(26), 429-457.

    Porting:
        Mirrors ``ts_fil_gabor.R``. Fidelity: exact (manual kernel + convolution,
        ~5e-9).
    """

    def __init__(self, window_size: float = 3, central_frequency: float = 0.1) -> None:
        if window_size <= 0:
            raise ValueError("Invalid window_size value")
        if not (0 <= central_frequency <= 0.5):
            raise ValueError("Invalid central_frequency value")
        self.window_size = window_size
        self.central_frequency = central_frequency

    def _kernel(self, radius: int) -> np.ndarray:
        t = np.arange(-radius, radius + 1)
        env = np.exp(-(t ** 2) / (2 * self.window_size ** 2))
        k = env * np.cos(2 * np.pi * self.central_frequency * t)
        ksum = k.sum()
        return k / ksum if abs(ksum) > np.finfo(float).eps else k

    def transform(self, data, **kwargs) -> np.ndarray:
        x = _vec(data)
        n = len(x)
        if n < 4 or np.sum(np.isfinite(x)) < 4:
            return x
        radius = min(max(1, int(np.ceil(3 * self.window_size))), n - 1)
        kernel = self._kernel(radius)
        left = x[radius:0:-1]                # rev(x[2:(radius+1)])
        right = x[n - 2:n - radius - 2:-1]   # rev(x[(n-radius):(n-1)])
        padded = np.concatenate([left, x, right])
        # centered convolution (R stats::filter sides=2) == correlation with the kernel
        conv = np.convolve(padded, kernel[::-1], mode="same")
        return conv[radius:radius + n]


class TsFilLowess(_Filter):
    """LOWESS smoothing (``ts_fil_lowess``).

    References:
        - Cleveland WS (1979). Robust locally weighted regression and smoothing
          scatterplots. JASA, 74(368), 829-836.

    Porting:
        Mirrors ``ts_fil_lowess.R``. Fidelity: approximate.
        - statsmodels ``lowess`` with ``delta=0.01*(n-1)``, ``it=3`` == R lowess
          (~5e-9).
    """

    def __init__(self, f: float = 0.2) -> None:
        self.f = f

    def transform(self, data, **kwargs) -> np.ndarray:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        y = _vec(data)
        x = np.arange(1, len(y) + 1)
        delta = 0.01 * (len(y) - 1)  # R lowess default delta
        return lowess(y, x, frac=self.f, it=3, delta=delta, return_sorted=False)


class _HoltWinters(_Filter):
    beta = False

    def __init__(self, gamma: bool = False) -> None:
        self.gamma = gamma

    def transform(self, data, **kwargs) -> np.ndarray:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        y = _vec(data)
        trend = "add" if self.beta else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(y, trend=trend, seasonal=None,
                                       initialization_method="estimated").fit()
        fitted = np.asarray(fit.fittedvalues, dtype=float)
        warmup = 2 if self.beta else 1  # mirror R HoltWinters warm-up length
        out = np.full(len(y), np.nan)
        out[warmup:] = fitted[warmup:]
        return out


class TsFilSes(_HoltWinters):
    """Simple exponential smoothing (``ts_fil_ses``).

    References:
        - Brown RG (1959). Statistical Forecasting for Inventory Control.

    Porting:
        Mirrors ``ts_fil_ses.R``. Fidelity: approximate (statsmodels vs R
        ``HoltWinters``; near-exact ~8e-5).
    """

    beta = False


class TsFilQes(_HoltWinters):
    """Double (trend) exponential smoothing (``ts_fil_qes``).

    References:
        - Winters PR (1960). Forecasting sales by exponentially weighted moving
          averages. Management Science, 6(3), 324-342.

    Porting:
        Mirrors ``ts_fil_qes.R``. Fidelity: approximate (HoltWinters-trend backend
        tolerance, ~0.13).
    """

    beta = True


class TsFilSeasAdj(_Filter):
    """Seasonal adjustment via STL (``ts_fil_seas_adj``).

    References:
        - Cleveland RB, Cleveland WS, McRae JE, Terpenning I (1990). STL: a
          seasonal-trend decomposition procedure based on loess. Journal of
          Official Statistics, 6(1), 3-73.

    Porting:
        Mirrors ``ts_fil_seas_adj.R``. Fidelity: approximate (R ``stl`` vs
        statsmodels ``STL`` backend tolerance, ~1.2).
    """

    def __init__(self, frequency: int | None = None) -> None:
        self.frequency = frequency

    @staticmethod
    def _estimate_frequency(x: np.ndarray) -> int:
        from statsmodels.tsa.stattools import acf
        max_lag = len(x) // 3
        if max_lag < 2:
            return 1
        vals = acf(x, nlags=max_lag, fft=False)[1:]
        lag = int(np.argmax(vals))
        if vals[lag] <= 0.3:
            return 1
        return lag + 1

    def transform(self, data, **kwargs) -> np.ndarray:
        from statsmodels.tsa.seasonal import STL
        x = _vec(data)
        freq = self.frequency if self.frequency is not None else self._estimate_frequency(x)
        if freq < 2 or len(x) < 2 * freq:
            return x
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = STL(x, period=freq, robust=True).fit()
        return x - np.asarray(res.seasonal, dtype=float)


class TsFilSpline(_Filter):
    """Cubic smoothing spline (``ts_fil_spline``).

    References:
        - Craven P, Wahba G (1978). Smoothing noisy data with spline functions.
          Numerische Mathematik, 31, 377-403.

    Porting:
        Mirrors ``ts_fil_spline.R``. Fidelity: approximate (R ``smooth.spline`` vs
        scipy ``make_smoothing_spline`` backend tolerance, ~0.14).
    """

    def __init__(self, spar: float | None = None) -> None:
        self.spar = spar

    def transform(self, data, **kwargs) -> np.ndarray:
        from scipy.interpolate import make_smoothing_spline
        x = _vec(data)
        n = len(x)
        if n < 4 or np.sum(np.isfinite(x)) < 4:
            return x
        t = np.arange(1, n + 1, dtype=float)
        valid = np.isfinite(x)
        try:
            lam = None if self.spar is None else float(self.spar)
            spline = make_smoothing_spline(t[valid], x[valid], lam=lam)
            result = spline(t)
        except Exception:  # noqa: BLE001 - fall back to the raw series on failure
            return x
        result[~np.isfinite(result)] = x[~np.isfinite(result)]
        return result


class TsFilKalman(_Filter):
    """Local-level Kalman filter smoothing (``ts_fil_kalman``; R uses KFAS).

    References:
        - Kalman RE (1960). A new approach to linear filtering and prediction
          problems. Journal of Basic Engineering, 82(1), 35-45.

    Porting:
        Mirrors ``ts_fil_kalman.R``. Fidelity: approximate.
        - Manual local-level KF (diffuse init P0=1e7) == R ``KFAS`` (~5e-9).
    """

    def __init__(self, H: float = 0.1, Q: float = 1.0) -> None:
        self.H = H
        self.Q = Q

    def transform(self, data, **kwargs) -> np.ndarray:
        y = _vec(data)
        n = len(y)
        att = np.empty(n)
        a, p = 0.0, 1e7  # diffuse initialization (as KFAS)
        for i in range(n):
            p_pred = p + self.Q
            f = p_pred + self.H
            k = p_pred / f
            a = a + k * (y[i] - a)
            p = p_pred * (1 - k)
            att[i] = a
        return att


def _decompose(y, engine, trials, noise_width, random_state):
    """Decompose ``y`` into IMFs (last row is the residual) via native EMD/EEMD."""
    if engine == "eemd":
        return eemd(y, trials=trials, noise_width=noise_width, random_state=random_state)
    if engine == "emd":
        return emd(y)
    raise ValueError(f"unknown EMD engine {engine!r} (use 'emd' or 'eemd')")


def _roughness(x: np.ndarray) -> float:
    """Normalized-second-difference roughness of an IMF (mirrors R ``fc_roughness``)."""
    x = x[~np.isnan(x)]
    if len(x) <= 2:
        return 0.0
    first_d = np.diff(x)
    scale = np.std(first_d, ddof=1)
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        return 0.0
    norm_first_d = (first_d - np.mean(first_d)) / scale
    return float(np.mean((np.diff(norm_first_d) ** 2) / 4))


class TsFilEmd(_Filter):
    """EMD denoising filter: remove the first IMF as noise (``ts_fil_emd``).

    Decomposes the series into Intrinsic Mode Functions and reconstructs it without
    the first (highest-frequency) IMF: ``result = data - imf1``.

    Parameters
    ----------
    engine:
        ``"emd"`` (default, single pass -- fastest) or ``"eemd"`` (noise-assisted
        ensemble, the analogue of the R ``CEEMD``).
    trials, noise_width, random_state:
        Used only when ``engine="eemd"`` (ensemble size, relative noise std, seed).

    References:
        - Huang, N. E., et al. (1998). The empirical mode decomposition and the
          Hilbert spectrum for nonlinear and non-stationary time series analysis.
          Proceedings of the Royal Society A, 454, 903-995.

    Porting:
        Replaces ``ts_fil_emd.R`` (which wrapped ``hht::CEEMD``) with the native
        :mod:`pytspredit.emd`. Fidelity: adapted (own dependency-free engine; EMD is
        stochastic/backend -- parity is quality-level, not value-wise). Native is
        canonical here and ~orders of magnitude faster than R ``hht``; the R package
        is to be backported to it (dropping ``hht``).
    """

    def __init__(self, engine: str = "emd", trials: int = 100,
                 noise_width: float = 0.05, random_state=None) -> None:
        self.engine = engine
        self.trials = trials
        self.noise_width = noise_width
        self.random_state = random_state

    def transform(self, data, **kwargs) -> np.ndarray:
        y = _vec(data)
        if len(y) < 4 or np.any(np.isnan(y)):
            return y
        imfs = _decompose(y, self.engine, self.trials, self.noise_width, self.random_state)
        if imfs.shape[0] < 2:
            return y
        return y - imfs[0]


class TsFilRemd(_Filter):
    """Robust EMD denoising filter (``ts_fil_remd``).

    Decomposes the series, scores each IMF's roughness, selects how many
    high-frequency IMFs to treat as noise from the cumulative-roughness curve (via
    :class:`pydaltoolbox.FitCurvatureMin`, capped by ``max_imfs``), and removes them.

    References:
        - Wu, Z., & Huang, N. E. (2009). Ensemble Empirical Mode Decomposition: a
          noise-assisted data analysis method. Advances in Adaptive Data Analysis,
          1(1), 1-41.

    Porting:
        Replaces ``ts_fil_remd.R`` (``hht::CEEMD`` + ``fit_curvature_min``) with the
        native :mod:`pytspredit.emd` + ``FitCurvatureMin``. Fidelity: adapted
        (dependency-free engine; stochastic/backend tolerance). Canonical here.
    """

    def __init__(self, engine: str = "emd", trials: int = 100, noise_width: float = 0.05,
                 max_imfs: int | None = None, random_state=None) -> None:
        self.engine = engine
        self.trials = trials
        self.noise_width = noise_width
        self.max_imfs = max_imfs
        self.random_state = random_state

    def _cutoff(self, cum_roughness: np.ndarray, nimf: int) -> int:
        cutoff = 1
        if nimf >= 3 and np.all(np.isfinite(cum_roughness)) and \
                np.sum(cum_roughness) > np.finfo(float).eps:
            from pydaltoolbox import FitCurvatureMin
            try:
                res = FitCurvatureMin().transform(cum_roughness)
                x = float(res["x"].iloc[0])
                if np.isfinite(x):
                    cutoff = int(x)
            except Exception:  # noqa: BLE001 - fall back to removing a single IMF
                cutoff = 1
        if self.max_imfs is not None:
            cutoff = min(cutoff, self.max_imfs)
        return max(0, min(cutoff, nimf - 1))

    def transform(self, data, **kwargs) -> np.ndarray:
        y = _vec(data)
        n = len(y)
        if n < 4 or np.any(np.isnan(y)):
            return y
        imfs = _decompose(y, self.engine, self.trials, self.noise_width, self.random_state)
        nimf = imfs.shape[0] - 1  # exclude the residual trend
        if nimf < 2:
            return y
        roughness = np.array([_roughness(imfs[k]) for k in range(nimf)])
        cutoff = self._cutoff(np.cumsum(roughness), nimf)
        if cutoff < 1:
            return y
        noise = imfs[:cutoff].sum(axis=0)
        return y - noise


class TsFilWavelet(_Filter):
    """Wavelet (MODWT) denoising filter (``ts_fil_wavelet``).

    Reconstructs the series with the finest-scale detail zeroed (``x - D1``). If
    ``filter`` is a list of names, :meth:`fit` selects the one maximizing R2 against
    the input (as in the R original).

    Parameters
    ----------
    filter:
        A wavelet filter name (``"haar"``, ``"d4"``, ``"la8"``, ``"bl14"``, ``"c6"``)
        or a list of names to select from in :meth:`fit`.

    References:
        - Mallat, S. (1989). A theory for multiresolution signal decomposition.
          IEEE TPAMI, 11(7), 674-693.
        - Percival, D. B., & Walden, A. T. (2000). Wavelet Methods for Time Series
          Analysis. Cambridge University Press.

    Porting:
        Replaces ``ts_fil_wavelet.R`` (``wavelets::modwt``/``imodwt``) with the native
        :mod:`pytspredit.wavelet` (pure numpy, no dependency). Fidelity: approximate
        (~5e-6 vs R; same algorithm/coefficients, different numeric path). Canonical
        here (the R package is to be backported to it, dropping ``wavelets``).
    """

    def __init__(self, filter: str | list[str] = "haar") -> None:
        self.filter = filter

    def fit(self, data=None, **kwargs) -> TsFilWavelet:
        if isinstance(self.filter, (list, tuple)):
            from .ts_reg import r2_ts
            x = _vec(data)
            best, best_r2 = None, -np.inf
            for name in self.filter:
                r2 = r2_ts(x, modwt_denoise(x, name))
                if r2 > best_r2:
                    best, best_r2 = name, r2
            self.filter = best
        return self

    def transform(self, data, **kwargs) -> np.ndarray:
        return modwt_denoise(_vec(data), self.filter)
