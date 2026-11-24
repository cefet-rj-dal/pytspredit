"""ARIMA regressor (``ts_arima.R``).

Univariate ARIMA built on ``statsmodels``. Unlike the sliding-window regressors,
``TsArima`` extends :class:`~pytspredit.TsReg` and operates on the raw series (a
``ts_data`` with ``sw = 1``).

Parity note: the R package uses ``forecast::Arima`` / ``forecast::auto.arima``.
``statsmodels`` is a different maximum-likelihood implementation, so forecasts are
close but not bit-identical (expected backend tolerance). With an explicit
``(p, d, q)`` the orders match exactly; ``auto`` order selection here is a compact
AICc search that approximates (but may not reproduce) ``auto.arima``'s choice.
"""
from __future__ import annotations

import warnings

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from .ts_reg import TsReg, _vec

__all__ = ["TsArima", "fit_arima"]


def _ndiffs(series, max_d: int = 2, alpha: float = 0.05) -> int:
    """Number of differences to reach stationarity via the KPSS test (like ndiffs)."""
    from statsmodels.tsa.stattools import kpss
    x = np.asarray(series, dtype=float)
    d = 0
    while d < max_d and len(x) >= 10:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pvalue = kpss(x, regression="c", nlags="auto")[1]
        except Exception:  # noqa: BLE001 - test can fail on degenerate series
            break
        if pvalue >= alpha:  # fail to reject stationarity -> stop
            break
        x = np.diff(x)
        d += 1
    return d


def _acf_pacf_candidates(series, d: int, max_order: int, base: int = 3):
    """Sparse (p, q) candidate sets from significant ACF/PACF lags (Box-Jenkins).

    Rather than a dense grid up to a fixed cap (auto.arima uses 5), the candidates are
    the small base orders ``{0..base}`` **plus the individually significant lags** of
    PACF (for p) and ACF (for q). So a seasonal spike (e.g. at lag 12) adds exactly that
    order to the search -- testing "with 12" when the data calls for it -- without the
    cost of a dense 0..12 grid. Lags are capped at ``max_order``.
    """
    from statsmodels.tsa.stattools import acf, pacf
    x = np.diff(series, n=d) if d > 0 else np.asarray(series, dtype=float)
    n = len(x)
    base_set = set(range(min(base, max_order) + 1))
    maxlag = int(min(max_order, n // 3, n - 2))
    if maxlag < 1:
        return sorted(base_set), sorted(base_set)
    band = 1.96 / np.sqrt(n)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = np.abs(acf(x, nlags=maxlag, fft=False)[1:])
            p = np.abs(pacf(x, nlags=maxlag)[1:])
    except Exception:  # noqa: BLE001 - fall back to the base orders on failure
        return sorted(base_set), sorted(base_set)

    def _top_lag(vals):
        # the single strongest significant lag beyond the base orders (e.g. seasonal 12)
        sig = [(vals[i], i + 1) for i in range(len(vals)) if vals[i] > band and i + 1 > base]
        return {max(sig)[1]} if sig else set()

    return sorted(base_set | _top_lag(p)), sorted(base_set | _top_lag(a))


def _smape(actual, pred) -> float:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    return float(np.mean(2.0 * np.abs(a - p) / (np.abs(a) + np.abs(p) + 1e-9)))


def _cv_error(series, order, trend, folds: int, horizon: int) -> float | None:
    """Rolling-origin out-of-sample error, matched to the forecast ``horizon``.

    Uses ``folds`` non-overlapping rolling origins, each forecasting ``horizon`` steps
    ahead and scoring mean sMAPE over the horizon. Matching the CV horizon to the actual
    forecast horizon is essential -- selecting by 1-step error picks orders that
    generalize poorly multi-step.
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    need = order[0] + order[1] + order[2] + 2
    k = min(folds, (n - need) // horizon)
    if k < 1:
        return None
    errs = []
    for i in range(k):
        cut = n - (k - i) * horizon
        if cut <= need:
            return None
        res = _try_fit(x[:cut], order, trend)
        if res is None:
            return None
        errs.append(_smape(x[cut:cut + horizon], np.asarray(res.forecast(horizon))))
    return float(np.mean(errs)) if errs else None


def _auto_order_cv(series, max_order: int = 12, folds: int = 2, horizon=None):
    """ARIMA++ order selection: ACF/PACF-driven candidates + horizon-matched CV.

    1. ``d`` from the KPSS test; 2. (p, q) candidates = small base orders plus the
    significant ACF/PACF lags (not a fixed cap of 5); 3. score each by rolling-origin
    out-of-sample sMAPE matched to the forecast ``horizon``; 4. pick the lowest.
    ``horizon`` defaults to the detected seasonal period (largest candidate lag), so
    seasonal series are selected for multi-step accuracy. Falls back to the compact
    AICc search if CV is not usable (very short series).
    """
    x = np.asarray(series, dtype=float)
    d = _ndiffs(x)
    trend = "c" if d == 0 else ("t" if d == 1 else "n")
    p_cands, q_cands = _acf_pacf_candidates(x, d, max_order)
    h = int(horizon) if horizon else max(2, max(p_cands + q_cands))
    scored = []  # (cv_error, order)
    for p in p_cands:
        for q in q_cands:
            if (p == 0 and q == 0) or p + q > max_order + 3:
                continue
            err = _cv_error(x, (p, d, q), trend, folds, h)
            if err is not None:
                scored.append((err, (p, d, q)))
    if not scored:  # CV unusable (very short series) -> compact AICc search
        return _auto_order(x)
    scored.sort(key=lambda t: t[0])
    best_order = scored[0][1]
    model = _try_fit(x, best_order, trend) or _fit_arima(x, best_order, trend)
    return best_order, trend, model


def _trend(d: int, include_mean: bool, include_drift: bool) -> str:
    if d == 0 and include_mean:
        return "c"
    if d == 1 and include_drift:
        return "t"
    return "n"


def _fit_arima(series, order, trend):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(series, order=order, trend=trend).fit()


def _try_fit(series, order, trend):
    """Fit an ARIMA, returning ``None`` if the order cannot be estimated."""
    try:
        return _fit_arima(series, order, trend)
    except Exception:  # noqa: BLE001 - many fit errors are possible; skip the order
        return None


def _auto_order(series):
    """Compact AICc search approximating auto.arima (p,q in 0..3, d in 0..2)."""
    best = None
    for d in (0, 1, 2):
        trend = "c" if d == 0 else ("t" if d == 1 else "n")
        for p in range(4):
            for q in range(4):
                if p == 0 and q == 0 and d == 0:
                    continue
                res = _try_fit(series, (p, d, q), trend)
                if res is None:
                    continue
                if best is None or res.aicc < best[0]:
                    best = (res.aicc, (p, d, q), trend, res)
    if best is None:
        return (1, 0, 0), "c", _fit_arima(series, (1, 0, 0), "c")
    return best[1], best[2], best[3]


def fit_arima(series, auto="aicc", max_order=12, cv_folds=2, horizon=None):
    """Automatic ARIMA fit -- the automatic order-selection entry point.

    Selects the ``(p, d, q)`` order automatically and returns the fitted model. This is
    the univariate auto-fit entry point used by :class:`TsArima` (constructor orders left
    ``None`` or set to ``(0, 0, 0)``), and is exported for standalone reuse. Both this
    function and its canonical R counterpart -- the exported ``fit_arima()`` in
    ``ts_arima.R`` -- play the role of ``forecast::auto.arima``.

    * ``auto="aicc"`` (default) -- compact AICc search (p, q in 0..3), approximating
      ``forecast::auto.arima``.
    * ``auto="cv"`` -- **ARIMA++**: ACF/PACF-driven candidates + horizon-matched
      rolling-origin cross-validation. ``max_order``/``cv_folds``/``horizon`` tune it.

    Returns ``(order, trend, model)``: the selected ``(p, d, q)`` tuple, the
    statsmodels trend string, and the fitted ``statsmodels`` results object.
    """
    x = _vec(series)
    if auto == "cv":
        return _auto_order_cv(x, max_order=max_order, folds=cv_folds, horizon=horizon)
    return _auto_order(x)


class TsArima(TsReg):
    """ARIMA(p, d, q) forecaster. Leave orders ``None`` (or ``(0, 0, 0)``) for auto.

    Theory:
        ARIMA(p, d, q) models a univariate series by combining ``p`` autoregressive
        terms, ``d`` orders of differencing to reach stationarity, and ``q`` moving-average
        terms (Box & Jenkins). Orders are either given explicitly or selected automatically
        (see below); parameters are estimated by maximum likelihood.

    Automatic order selection (when ``p``/``d``/``q`` are ``None`` or all ``0`` --
    ``(0, 0, 0)`` is read as the auto-fit sentinel, so a pure white-noise model is not
    reachable) delegates to the exported :func:`fit_arima`:

    * ``auto="aicc"`` (default) -- a compact AICc search (p,q in 0..3), approximating
      ``forecast::auto.arima``.
    * ``auto="cv"`` -- **ARIMA++**: a data-driven alternative that addresses two known
      weaknesses of ``auto.arima``. (1) The (p, q) search bounds come from the
      **significant ACF/PACF lags** (Box-Jenkins identification) rather than a fixed
      cap of 5, so a seasonal spike (e.g. at lag 12) extends the search to cover it.
      (2) It searches the **full grid** (not greedy stepwise) and selects the order by
      **rolling-origin cross-validated forecast error** (out-of-sample sMAPE), not by
      the in-sample AICc proxy. ``max_order`` caps the bounds; ``cv_folds`` sets the
      number of rolling origins.

    References:
        - Box GEP, Jenkins GM, Reinsel GC, Ljung GM (2015). Time Series Analysis:
          Forecasting and Control. Wiley (ACF/PACF identification).
        - Hyndman RJ, Khandakar Y (2008). Automatic time series forecasting: the
          forecast package for R. Journal of Statistical Software, 27(3), 1-22.
          doi:10.18637/jss.v027.i03
        - Tashman LJ (2000). Out-of-sample tests of forecasting accuracy: an analysis
          and review. International Journal of Forecasting, 16(4), 437-450
          (rolling-origin evaluation).

    Porting:
        Mirrors ``ts_arima.R``. Fidelity: approximate (explicit ``(p,d,q)``) /
        adapted (``auto="cv"`` is a new, improved selector, not in R's ``auto.arima``).
        - Built on ``statsmodels`` (R used ``forecast::Arima``/``auto.arima``):
          explicit ``(p,d,q)`` matches exactly; ``auto="aicc"`` approximates
          ``auto.arima`` (different MLE backend, ~6e-4); ``auto="cv"`` (ARIMA++) is an
          algorithmic improvement validated against ``auto.arima`` on real benchmark data
          (NN5: ~14% lower sMAPE; see ``develop/parity.md`` Section 5.3).
    """

    def __init__(self, p=None, d=None, q=None, auto="aicc",
                 max_order=12, cv_folds=2, horizon=None) -> None:
        super().__init__()
        self.p, self.d, self.q = p, d, q
        self.auto = auto                 # "aicc" (compact) or "cv" (ARIMA++)
        self.max_order = max_order
        self.cv_folds = cv_folds
        self.horizon = horizon           # CV horizon for auto="cv" (defaults to season)
        specified = p is not None and d is not None and q is not None
        auto_zero = specified and p == 0 and d == 0 and q == 0  # (0,0,0) = auto-fit
        self.manual_order = specified and not auto_zero
        self.include_mean = bool(self.manual_order and d == 0)
        self.include_drift = bool(self.manual_order and d == 1)
        self.model = None
        self.train = None

    def fit(self, x, y=None, **kwargs) -> TsArima:
        series = _vec(x)
        self.train = series
        if self.manual_order:
            trend = _trend(self.d, self.include_mean, self.include_drift)
            self.model = _fit_arima(series, (self.p, self.d, self.q), trend)
        else:
            (self.p, self.d, self.q), _, self.model = fit_arima(
                series, auto=self.auto, max_order=self.max_order,
                cv_folds=self.cv_folds, horizon=self.horizon)
            self.include_mean = self.d == 0
            self.include_drift = self.d == 1
        return self

    def _order(self):
        return (self.p, self.d, self.q)

    def predict(self, x, y=None, steps_ahead=None, **kwargs) -> np.ndarray:
        xv = _vec(x)
        # in-sample fit when x is the training series
        if len(xv) == len(self.train) and np.allclose(xv, self.train):
            return np.asarray(self.model.fittedvalues, dtype=float)

        if steps_ahead is None:
            steps_ahead = len(xv)

        if steps_ahead == 1 and len(xv) != 1:
            # rolling one-step-ahead: refit with known order as actuals arrive
            trend = _trend(self.d, self.include_mean, self.include_drift)
            preds = []
            history = list(self.train)
            model = self.model
            for i in range(len(xv)):
                preds.append(float(np.asarray(model.forecast(1))[0]))
                history.append(xv[i])
                refit = _try_fit(np.asarray(history, dtype=float), self._order(), trend)
                if refit is None:  # fall back to auto selection, as in ts_arima.R
                    (self.p, self.d, self.q), trend, refit = _auto_order(
                        np.asarray(history, dtype=float))
                model = refit
            return np.asarray(preds, dtype=float)

        return np.asarray(self.model.forecast(steps_ahead), dtype=float)
