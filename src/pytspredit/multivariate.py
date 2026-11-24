"""Multivariate (target-centered) time series modeling.

Python counterpart of the multivariate branch of ``tspredit``:
``ts_data_mv.R``, ``ts_regsw_mv.R``, ``ts_reg_mv.R``, ``ts_lm_mv.R``,
``ts_var.R`` and ``ts_arimax.R``.

The whole branch stays *target-centered*: a multivariate object always declares
one distinguished target variable ``y`` while the auxiliary variables
``x1, ..., xn`` either help forecast ``y`` or are forecast by their own
pipelines. ``predict`` therefore returns the forecast of ``y`` (as an ndarray),
with the full system forecast attached as attributes (:class:`TsMvPrediction`).

Two orchestration styles mirror the univariate split:

* :class:`TsRegswMv` -- the *sliding-window* orchestrator: one univariate
  learner per variable (each its own :func:`ts_mv_spec`), coordinated over a
  shared ``window_size`` and forecast recursively. This reuses the existing
  univariate regressors (:class:`~pytspredit.TsMlp`, ...) as polymorphic blocks.
* :class:`TsRegMv` -- the *singular* (aligned, ``sw = 1``) branch for models that
  consume synchronized observations directly: :class:`TsLmMv` (linear
  regression), :class:`TsVar` (vector autoregression) and :class:`TsArimax`
  (ARIMA with external regressors).

References:
    - Lütkepohl H (2005). New Introduction to Multiple Time Series Analysis.
      Springer. (VAR)
    - Tsay RS (2014). Multivariate Time Series Analysis with R and Financial
      Applications. Wiley.
    - Box GEP, Jenkins GM, Reinsel GC, Ljung GM (2015). Time Series Analysis:
      Forecasting and Control. Wiley. (ARIMAX)
    - Hyndman RJ, Athanasopoulos G (2021). Forecasting: Principles and Practice,
      3rd ed. OTexts. https://otexts.com/fpp3/
    - Montgomery DC, Peck EA, Vining GG (2021). Introduction to Linear
      Regression Analysis. Wiley. (ts_lm_mv)

Porting:
    Mirrors ``ts_data_mv.R``, ``ts_regsw_mv.R``, ``ts_reg_mv.R``, ``ts_lm_mv.R``,
    ``ts_var.R`` and ``ts_arimax.R``. Fidelity: approximate.
    - ``TsDataMv`` is a ``pandas.DataFrame`` subclass carrying the ``y``/``x``/
      ``variables``/``lags``/``sw``/``representation`` metadata that the R object
      keeps as attributes; the univariate ``ts_data`` stays a plain DataFrame,
      but the multivariate workflow needs the metadata to travel with the object.
    - ``TsLmMv`` reuses :class:`~pytspredit.darima_warma._RLinear` (R ``lm``
      column-dropping on rank-deficient designs) instead of ``sklearn``
      min-norm, so it matches R's ``stats::lm`` exactly.
    - ``TsVar`` is a native OLS VAR with the same AICc order search as R.
    - ``TsArimax`` wraps ``statsmodels`` ARIMA (with ``exog``) where R used
      ``forecast::Arima``/``auto.arima``; explicit ``(p, d, q)`` matches, the
      automatic search approximates ``auto.arima`` (MLE backend differs).
"""
from __future__ import annotations

import warnings
from typing import ClassVar

import numpy as np
import pandas as pd

from .arima import _trend
from .darima_warma import _RLinear
from .lagmap import TsLagmap
from .ts_reg import TsReg

__all__ = [
    "TsArimax",
    "TsDataMv",
    "TsLmMv",
    "TsMvPrediction",
    "TsMvSpec",
    "TsRegMv",
    "TsRegswMv",
    "TsVar",
    "adjust_ts_data_mv",
    "ts_data_mv",
    "ts_mv_spec",
]


# ===========================================================================
# ts_data_mv.R -- multivariate data container
# ===========================================================================

class TsDataMv(pd.DataFrame):
    """Multivariate time-series container (``ts_data_mv``).

    A :class:`pandas.DataFrame` that additionally carries the multivariate
    metadata (target name, auxiliary names, variable order, per-variable lags,
    window size and representation) so it can travel through sampling and
    modeling. Build one with :func:`ts_data_mv`.
    """

    _metadata: ClassVar[list[str]] = [
        "mv_y", "mv_x", "mv_variables", "mv_lags", "mv_sw", "mv_representation",
    ]

    @property
    def _constructor(self):
        return TsDataMv


def _meta(obj, key, default=None):
    return getattr(obj, key, default)


def adjust_ts_data_mv(data, y, x=None, sw=1, variables=None, lags=None,
                      representation="aligned") -> TsDataMv:
    """Restore multivariate metadata after subsetting (``adjust_ts_data_mv``).

    Mirrors the univariate :func:`~pytspredit.adjust_ts_data`: it re-attaches the
    target/auxiliary names and the ``"aligned"`` vs ``"windowed"`` representation.
    """
    if representation not in ("aligned", "windowed"):
        raise ValueError('representation must be "aligned" or "windowed".')
    df = pd.DataFrame(data).copy()

    if variables is None:
        variables = [y] + (list(x) if x is not None else [])
    else:
        variables = list(variables)
    if x is None:
        x = [v for v in variables if v != y]
    else:
        x = list(x)

    if representation == "aligned":
        df = df[[y, *x]]

    out = TsDataMv(df)
    out.mv_y = y
    out.mv_x = x
    out.mv_variables = variables
    out.mv_lags = lags
    out.mv_sw = int(sw)
    out.mv_representation = representation
    return out


def ts_data_mv(data, y=None, x=None, sw=1, variables=None, lags=None,
               transforms=None) -> TsDataMv:
    """Construct a multivariate time-series object (``ts_data_mv``).

    With ``sw = 1`` it stores *aligned* multivariate observations (one row per
    time instant, one column per variable). With ``sw > 1`` it materializes
    *lagged* multivariate windows (columns named ``var_tk``), optionally
    restricted to given ``variables``/``lags`` and with per-variable raw-series
    ``transforms`` applied first.

    Parameters
    ----------
    data:
        A :class:`pandas.DataFrame` (one column per variable), or an existing
        :class:`TsDataMv` whose stored metadata is reused by default.
    y:
        Name of the target variable.
    x:
        Auxiliary variable names (default: all columns except ``y``).
    sw:
        Temporal width. ``1`` = aligned, ``> 1`` = lagged windows.
    variables, lags, transforms:
        Optional multivariate feature space for ``sw > 1`` (see module docstring).
    """
    if isinstance(data, TsDataMv):
        if y is None:
            y = _meta(data, "mv_y")
        if x is None:
            x = _meta(data, "mv_x")
    df = pd.DataFrame(data).copy()

    if y is None or not isinstance(y, str) or y not in df.columns:
        raise ValueError("y must be the name of a column in data.")
    if not isinstance(sw, (int, np.integer)) or sw < 1:
        raise ValueError("sw must be a positive integer.")
    sw = int(sw)

    if x is None:
        x = [c for c in df.columns if c != y]
    else:
        x = list(x)
    if any(c not in df.columns for c in x):
        raise ValueError("All x variables must be present in data.")
    raw_variables = [y, *x]
    if len(set(raw_variables)) != len(raw_variables):
        raise ValueError("y and x must not contain duplicated variable names.")
    if not all(np.issubdtype(df[c].dtype, np.number) for c in raw_variables):
        raise ValueError("All variables in ts_data_mv must be numeric.")

    df = df[raw_variables]

    if sw == 1 and variables is None and lags is None and transforms is None:
        return adjust_ts_data_mv(df, y=y, x=x, sw=1, representation="aligned")

    selected = list(variables) if variables is not None else raw_variables
    spec = _mv_as_spec(ts_mv_spec(TsReg(), variables=selected, lags=lags,
                                  transforms=transforms),
                       default_variables=selected,
                       allowed_variables=raw_variables, window_size=sw)

    prepared = _mv_prepare_data_for_spec(df, spec)
    max_lag = max(max(v) for v in spec.lags.values())
    n = prepared.shape[0]
    if n <= max_lag:
        raise ValueError("Not enough observations to build the requested window.")
    origins = np.arange(max_lag, n)
    windows = _mv_build_input_matrix(prepared, spec, origins)
    return adjust_ts_data_mv(windows, y=y, x=[v for v in selected if v != y],
                             sw=sw, variables=selected, lags=spec.lags,
                             representation="windowed")


# ===========================================================================
# ts_regsw_mv.R -- spec object + orchestration helpers
# ===========================================================================

class TsMvSpec:
    """Per-variable submodel specification (``ts_mv_spec``).

    Declares the learner (``model``) for one variable plus which ``variables``
    feed it, which ``lags`` of each are used, and optional per-variable
    ``transforms`` applied to the raw series before windows are built.
    """

    def __init__(self, model, variables=None, lags=None, transforms=None) -> None:
        self.model = model
        self.variables = variables
        self.lags = lags
        self.transforms = transforms


def ts_mv_spec(model, variables=None, lags=None, transforms=None) -> TsMvSpec:
    """Wrap a univariate model for use inside the multivariate orchestrator."""
    return TsMvSpec(model, variables=variables, lags=lags, transforms=transforms)


def _mv_default_lags(window_size):
    return list(range(window_size))


def _mv_as_spec(spec, default_variables, allowed_variables, window_size) -> TsMvSpec:
    if not isinstance(spec, TsMvSpec):
        spec = ts_mv_spec(spec)
    spec = TsMvSpec(spec.model, spec.variables, spec.lags, spec.transforms)

    if spec.variables is None:
        spec.variables = list(default_variables)
    else:
        spec.variables = list(spec.variables)
    extra = [v for v in spec.variables if v not in allowed_variables]
    if extra:
        raise ValueError(f"Unknown variables in multivariate specification: {extra}")

    if spec.lags is None:
        spec.lags = {v: _mv_default_lags(window_size) for v in spec.variables}
    else:
        lags = dict(spec.lags)
        for v in spec.variables:
            if v not in lags:
                lags[v] = _mv_default_lags(window_size)
        spec.lags = {v: lags[v] for v in spec.variables}

    cleaned = {}
    for var, values in spec.lags.items():
        vals = sorted({int(v) for v in np.asarray(values).ravel() if np.isfinite(v)},
                      reverse=True)
        if len(vals) == 0 or any(v < 0 for v in vals):
            raise ValueError("Lag definitions must contain non-negative integers.")
        cleaned[var] = vals
    spec.lags = cleaned

    spec.transforms = _mv_normalize_transforms(spec.transforms, spec.variables)
    return spec


def _mv_normalize_transforms(transforms, variables):
    if transforms is None:
        return {v: [] for v in variables}
    if not isinstance(transforms, dict):
        raise ValueError(  # noqa: TRY004
            "transforms must be None or a dict keyed by variable name.")
    unknown = [k for k in transforms if k not in variables]
    if unknown:
        raise ValueError(f"Unknown variables in transforms: {unknown}")
    normalized = {}
    for var in variables:
        entry = transforms.get(var)
        if entry is None:
            normalized[var] = []
        elif isinstance(entry, (list, tuple)):
            normalized[var] = list(entry)
        else:
            normalized[var] = [entry]
    return normalized


def _mv_apply_transform_sequence(series, transforms):
    result = np.asarray(series, dtype=float)
    for transformer in transforms:
        transformer.fit(result)
        result = np.asarray(transformer.transform(result), dtype=float)
    return result


def _mv_prepare_data_for_spec(data, spec):
    prepared = pd.DataFrame(data).copy()
    for var in spec.variables:
        prepared[var] = _mv_apply_transform_sequence(prepared[var].to_numpy(dtype=float),
                                                      spec.transforms[var])
    return prepared


def _mv_build_input_matrix(data, spec, origins):
    origins = np.asarray(origins, dtype=int)
    blocks = {}
    for var in spec.variables:
        values = data[var].to_numpy(dtype=float)
        for lag in spec.lags[var]:
            blocks[f"{var}_t{lag}"] = values[origins - lag]
    return pd.DataFrame(blocks)


def _mv_prepare_supervised(data, target, spec):
    prepared = _mv_prepare_data_for_spec(data, spec)
    max_lag = max(max(v) for v in spec.lags.values())
    n = prepared.shape[0]
    if n <= max_lag + 1:
        raise ValueError("Not enough observations to build the requested window.")
    origins = np.arange(max_lag, n - 1)
    input_ = _mv_build_input_matrix(prepared, spec, origins)
    output = data[target].to_numpy(dtype=float)[origins + 1]
    keep = np.isfinite(input_.to_numpy(dtype=float)).all(axis=1) & np.isfinite(output)
    return input_.loc[keep].reset_index(drop=True), output[keep]


def _mv_prepare_one_step_input(data, spec):
    prepared = _mv_prepare_data_for_spec(data, spec)
    origin = np.array([data.shape[0] - 1])
    return _mv_build_input_matrix(prepared, spec, origin)


class TsMvPrediction(np.ndarray):
    """Target forecast (``ts_mv_prediction``) with the system forecast attached.

    Behaves as the numeric forecast of the target ``y``; the whole-system
    forecast is available via :attr:`system` (a DataFrame) plus the
    ``y_name``/``x_names``/``variables``/``prediction_x`` attributes.
    """

    def __new__(cls, values):
        obj = np.asarray(values, dtype=float).view(cls)
        obj.y_name = None
        obj.x_names = None
        obj.variables = None
        obj.prediction_x = None
        obj.system = None
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        for attr in ("y_name", "x_names", "variables", "prediction_x", "system"):
            setattr(self, attr, getattr(obj, attr, None))


def _mv_compose_prediction(y_name, x_names, variables, prediction_y, prediction_x):
    prediction_y = np.asarray(prediction_y, dtype=float)
    prediction_x = {k: np.asarray(v, dtype=float) for k, v in prediction_x.items()}
    system = pd.DataFrame({y_name: prediction_y, **prediction_x})[[y_name, *x_names]]

    out = TsMvPrediction(prediction_y)
    out.y_name = y_name
    out.x_names = list(x_names)
    out.variables = [y_name, *x_names]
    out.prediction_x = prediction_x
    out.system = system
    return out


def _mv_fit_submodel(model, input_, output, series=None):
    from .ts_reg import TsRegSW
    if isinstance(model, TsRegSW):
        model.input_size = input_.shape[1]
        model.input_map = TsLagmap("recent")
        return model.fit(input_, output)
    if series is not None:
        return model.fit(series)
    raise ValueError("This model type is not supported in multivariate mode.")


def _mv_predict_submodel(template, fitted, spec, history, target):
    from .ts_reg import TsRegSW
    if isinstance(fitted, TsRegSW):
        input_ = _mv_prepare_one_step_input(history, spec)
        return float(np.asarray(fitted.predict(input_, steps_ahead=1)).ravel()[0])
    series = history[target].to_numpy(dtype=float)
    fitted_step = template.fit(series)
    pred = fitted_step.predict(series[-1:], steps_ahead=1)
    return float(np.asarray(pred).ravel()[0])


# ===========================================================================
# ts_regsw_mv.R -- sliding-window multivariate orchestrator
# ===========================================================================

class TsRegswMv(TsReg):
    """Multivariate sliding-window orchestrator (``ts_regsw_mv``).

    Coordinates one target model (``model_y``, a :class:`TsMvSpec` over a
    sliding-window learner) and one auxiliary model per covariate (``models_x``),
    forecasting the whole system recursively while staying target-centered.

    Parameters
    ----------
    model_y:
        :class:`TsMvSpec` (or bare model) for the target. Its learner must be a
        sliding-window regressor (:class:`~pytspredit.TsRegSW`).
    models_x:
        ``dict`` mapping each auxiliary variable to its :class:`TsMvSpec`.
    window_size:
        Base temporal memory available to every variable.
    """

    def __init__(self, model_y, models_x, window_size: int = 30) -> None:
        super().__init__()
        if model_y is None:
            raise ValueError("model_y must be provided.")
        if not models_x:
            raise ValueError("models_x must provide one model per auxiliary variable.")
        if not isinstance(window_size, (int, np.integer)) or window_size < 1:
            raise ValueError("window_size must be a positive integer.")
        self.model_y = model_y
        self.models_x = dict(models_x)
        self.window_size = int(window_size)
        self.fitted_models_x = {}
        self.fitted_model_y = None
        self.history = None
        self.y_name = None
        self.x_names = None
        self.variables = None

    def _validate_specs(self, data):
        from .ts_reg import TsRegSW
        y_name = _meta(data, "mv_y")
        x_names = _meta(data, "mv_x")
        variables = _meta(data, "mv_variables")
        if set(self.models_x) != set(x_names):
            raise ValueError("models_x must contain exactly one model per auxiliary "
                             "variable in ts_data_mv.")
        self.model_y = _mv_as_spec(self.model_y, default_variables=variables,
                                   allowed_variables=variables,
                                   window_size=self.window_size)
        self.models_x = {
            name: _mv_as_spec(self.models_x[name], default_variables=[name],
                              allowed_variables=variables,
                              window_size=self.window_size)
            for name in x_names
        }
        if not isinstance(self.model_y.model, TsRegSW):
            raise ValueError(  # noqa: TRY004
                "The target model in ts_regsw_mv must be a sliding-window regressor "
                "(TsRegSW).")
        for name in x_names:
            spec = self.models_x[name]
            if not isinstance(spec.model, TsReg):
                raise ValueError(  # noqa: TRY004
                    "All auxiliary models must inherit from TsReg.")
            if not isinstance(spec.model, TsRegSW) and spec.variables != [name]:
                raise ValueError("Raw-series auxiliary models can only use their own "
                                 "variable as input.")
        self.y_name = y_name
        self.x_names = list(x_names)
        self.variables = list(variables)
        return self

    def fit(self, x, y=None, **kwargs) -> TsRegswMv:
        from .ts_reg import TsRegSW
        if not isinstance(x, TsDataMv) or _meta(x, "mv_representation") != "aligned":
            raise ValueError("fit expects aligned ts_data_mv data (sw = 1).")
        self._validate_specs(x)
        data = pd.DataFrame(x)

        self.fitted_models_x = {}
        for name in self.x_names:
            spec = self.models_x[name]
            if isinstance(spec.model, TsRegSW):
                inp, out = _mv_prepare_supervised(data, name, spec)
                self.fitted_models_x[name] = _mv_fit_submodel(spec.model, inp, out)
            else:
                self.fitted_models_x[name] = _mv_fit_submodel(
                    spec.model, None, None, series=data[name].to_numpy(dtype=float))

        inp_y, out_y = _mv_prepare_supervised(data, self.y_name, self.model_y)
        self.fitted_model_y = _mv_fit_submodel(self.model_y.model, inp_y, out_y)

        self.history = adjust_ts_data_mv(data, y=self.y_name, x=self.x_names)
        return self

    def predict(self, x=None, steps_ahead: int = 1, return_all: bool = False,
                **kwargs) -> TsMvPrediction:
        if x is None:
            if self.history is None:
                raise ValueError("This model has no stored history. Fit it or provide x.")
            history = self.history
        else:
            if not isinstance(x, TsDataMv) or _meta(x, "mv_representation") != "aligned":
                raise ValueError("predict expects x to be None or aligned ts_data_mv.")
            history = x
        history = adjust_ts_data_mv(pd.DataFrame(history), y=self.y_name, x=self.x_names)
        steps_ahead = int(steps_ahead)

        prediction_y = np.zeros(steps_ahead)
        prediction_x = {name: np.zeros(steps_ahead) for name in self.x_names}

        for step in range(steps_ahead):
            next_row = {v: np.nan for v in self.variables}
            for name in self.x_names:
                spec = self.models_x[name]
                next_row[name] = _mv_predict_submodel(
                    spec.model, self.fitted_models_x[name], spec, history, name)
                prediction_x[name][step] = next_row[name]
            next_row[self.y_name] = _mv_predict_submodel(
                self.model_y.model, self.fitted_model_y, self.model_y, history, self.y_name)
            prediction_y[step] = next_row[self.y_name]

            new = pd.DataFrame([[next_row[v] for v in self.variables]], columns=self.variables)
            history = adjust_ts_data_mv(pd.concat([pd.DataFrame(history), new],
                                                  ignore_index=True),
                                        y=self.y_name, x=self.x_names)

        self.history = history
        return _mv_compose_prediction(self.y_name, self.x_names, self.variables,
                                      prediction_y, prediction_x)


# ===========================================================================
# ts_reg_mv.R -- singular (aligned) multivariate base
# ===========================================================================

class TsRegMv(TsReg):
    """Singular multivariate regression base (``ts_reg_mv``).

    Base class for models that consume aligned multivariate observations
    (``sw = 1``) directly -- :class:`TsLmMv`, :class:`TsVar`, :class:`TsArimax`.
    It stays target-centered: it declares the target ``y`` and, when auxiliary
    future values are unknown, forecasts them with the univariate models in
    ``models_x``.
    """

    def __init__(self, models_x=None) -> None:
        super().__init__()
        self.models_x = models_x
        self.fitted_models_x = {}
        self.history = None
        self.y_name = None
        self.x_names = None
        self.variables = None
        self.model = None

    # -- shared helpers (reg_mv_* in R) --------------------------------------
    @staticmethod
    def _validate_data(x):
        if not isinstance(x, TsDataMv):
            raise ValueError("This method expects a ts_data_mv object.")  # noqa: TRY004
        if _meta(x, "mv_representation") != "aligned":
            raise ValueError("This method expects aligned ts_data_mv data (sw = 1).")
        return x

    def _set_metadata(self, data):
        self.y_name = _meta(data, "mv_y")
        self.x_names = list(_meta(data, "mv_x"))
        self.variables = list(_meta(data, "mv_variables"))
        self.history = adjust_ts_data_mv(pd.DataFrame(data), y=self.y_name, x=self.x_names)
        return self

    def _validate_models_x(self):
        if self.models_x is None:
            self.models_x = {}
        if len(self.models_x) == 0:
            return self
        if set(self.models_x) != set(self.x_names):
            raise ValueError("models_x must contain exactly one model per auxiliary "
                             "variable.")
        for model in self.models_x.values():
            if not isinstance(model, TsReg):
                raise ValueError(  # noqa: TRY004
                    "All auxiliary models in models_x must inherit from TsReg.")
        return self

    def _fit_aux_models(self, data):
        self.fitted_models_x = {}
        for name in self.x_names:
            if self.models_x.get(name) is not None:
                self.fitted_models_x[name] = self.models_x[name].fit(
                    data[name].to_numpy(dtype=float))
        return self

    def _forecast_aux(self, x=None, steps_ahead=1):
        if x is not None:
            future = pd.DataFrame(x)
            if any(c not in future.columns for c in self.x_names):
                raise ValueError("Provided future data must contain all auxiliary variables.")
            future = future[self.x_names]
            if future.shape[0] < steps_ahead:
                raise ValueError("Provided future data has fewer rows than steps_ahead.")
            return future.iloc[:steps_ahead].reset_index(drop=True)
        if len(self.fitted_models_x) == 0:
            raise ValueError("Future auxiliary values are required. Provide x or fit models_x.")
        cols = {}
        for name in self.x_names:
            pred = self.fitted_models_x[name].predict(None, steps_ahead=steps_ahead)
            cols[name] = np.asarray(pred, dtype=float).ravel()
        return pd.DataFrame(cols)[self.x_names]

    def _update_history(self, future_x, prediction_y):
        next_rows = future_x.copy()
        next_rows[self.y_name] = np.asarray(prediction_y, dtype=float)
        next_rows = next_rows[self.variables]
        history = pd.concat([pd.DataFrame(self.history), next_rows], ignore_index=True)
        return adjust_ts_data_mv(history, y=self.y_name, x=self.x_names)


# ===========================================================================
# ts_lm_mv.R -- multivariate linear regression
# ===========================================================================

class TsLmMv(TsRegMv):
    """Target-centered multivariate linear regression (``ts_lm_mv``).

    Regresses the target on the synchronized auxiliary variables. When future
    auxiliary values are unknown they are generated by the univariate models in
    ``models_x``.

    Parameters
    ----------
    models_x:
        Optional ``dict`` of one univariate model per auxiliary variable.
    formula:
        Optional ``"y ~ x1 + x2"`` string; the right-hand-side names pick the
        features. When omitted, the target is regressed on ``features`` (or on
        all auxiliary variables).
    features:
        Optional explicit feature-name list used when ``formula`` is ``None``.

    References:
        - Montgomery DC, Peck EA, Vining GG (2021). Introduction to Linear
          Regression Analysis. Wiley.

    Porting:
        Mirrors ``ts_lm_mv.R``. Fidelity: approximate.
        - Uses :class:`~pytspredit.darima_warma._RLinear` (R ``lm``
          column-dropping) rather than ``sklearn`` min-norm, so it matches
          ``stats::lm`` on collinear/rank-deficient designs.
        - R formulas are not evaluated; a ``"y ~ x1 + x2"`` string is parsed only
          to extract the feature names on its right-hand side.
    """

    def __init__(self, models_x=None, formula=None, features=None) -> None:
        super().__init__(models_x=models_x)
        self.formula = formula
        self.features = features
        self.feature_names = None

    def fit(self, x, y=None, **kwargs) -> TsLmMv:
        x = self._validate_data(x)
        self._set_metadata(x)
        self._validate_models_x()
        self._fit_aux_models(pd.DataFrame(x))
        data = pd.DataFrame(x)

        if self.formula is not None:
            rhs = self.formula.split("~", 1)[1]
            features = [tok.strip() for tok in rhs.split("+") if tok.strip()]
        elif self.features is not None:
            features = list(self.features)
        else:
            features = list(self.x_names)
        self.feature_names = [f for f in features if f in data.columns]

        xd = data[self.feature_names].to_numpy(dtype=float)
        yd = data[self.y_name].to_numpy(dtype=float)
        self.model = _RLinear(intercept=True).fit(xd, yd)
        return self

    def predict(self, x=None, steps_ahead: int = 1, return_all: bool = False,
                **kwargs) -> TsMvPrediction:
        steps_ahead = int(steps_ahead)
        future_x = self._forecast_aux(x=x, steps_ahead=steps_ahead)
        prediction_y = self.model.predict(future_x[self.feature_names].to_numpy(dtype=float))
        self.history = self._update_history(future_x, prediction_y)
        prediction_x = {name: future_x[name].to_numpy(dtype=float) for name in self.x_names}
        return _mv_compose_prediction(self.y_name, self.x_names, self.variables,
                                      prediction_y, prediction_x)


# ===========================================================================
# ts_var.R -- vector autoregression (native OLS)
# ===========================================================================

def _var_build_design(data, variables, p, intercept=True):
    arr = data[variables].to_numpy(dtype=float)
    n = arr.shape[0]
    if n <= p:
        raise ValueError("Not enough observations for the requested VAR order.")
    y = arr[p:n, :]
    blocks = []
    if intercept:
        blocks.append(np.ones((n - p, 1)))
    for lag in range(1, p + 1):
        blocks.append(arr[p - lag:n - lag, :])
    x = np.hstack(blocks)
    return x, y


def _var_fit_order(data, variables, p, intercept=True):
    x, y = _var_build_design(data, variables, p, intercept=intercept)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    coef = np.nan_to_num(coef, nan=0.0)
    resid = y - x @ coef
    nobs, m = y.shape
    sigma = resid.T @ resid / nobs
    sign, logabsdet = np.linalg.slogdet(sigma)
    logdet = logabsdet if sign > 0 else np.inf
    loglik = -(nobs * m / 2) * (np.log(2 * np.pi) + 1) - (nobs / 2) * logdet
    kparams = x.shape[1] * m
    aic = -2 * loglik + 2 * kparams
    if (nobs - kparams - 1) > 0:
        aicc = aic + (2 * kparams * (kparams + 1)) / (nobs - kparams - 1)
    else:
        aicc = np.inf
    return {"coefficients": coef, "aicc": aicc, "p": p, "intercept": intercept}


def _var_select_order(data, variables, p_max, intercept=True):
    best = None
    for p in range(1, max(1, int(p_max)) + 1):
        try:
            fit = _var_fit_order(data, variables, p, intercept=intercept)
        except Exception:  # noqa: BLE001, S112 - skip orders that cannot be fitted
            continue
        if best is None or fit["aicc"] < best["aicc"]:
            best = fit
    if best is None:
        raise ValueError("Unable to fit any candidate VAR order.")
    return best


class TsVar(TsRegMv):
    """Vector autoregression over aligned observations (``ts_var``).

    Learns the whole multivariate system jointly by OLS over lagged observations,
    choosing the lag order by AICc when ``p`` is ``None``. The interface stays
    target-centered (``predict`` returns the target by default; the full system
    forecast is attached).

    Parameters
    ----------
    target:
        Target variable (default: the ``y`` of the ``ts_data_mv``).
    p:
        Lag order; ``None`` selects it automatically over ``1..p_max``.
    p_max:
        Maximum order considered in the automatic search.
    intercept:
        Whether each equation includes an intercept.

    References:
        - Lütkepohl H (2005). New Introduction to Multiple Time Series Analysis.
          Springer.
        - Tsay RS (2014). Multivariate Time Series Analysis with R and Financial
          Applications. Wiley.

    Porting:
        Mirrors ``ts_var.R``. Fidelity: exact.
        - OLS VAR (``numpy.linalg.lstsq``) with the same AICc order search as R
          (R used ``stats::lm.fit``); results match to numerical precision (~1e-6).
    """

    def __init__(self, target=None, p=None, p_max: int = 5, intercept: bool = True) -> None:
        super().__init__(models_x=None)
        self.target = target
        self.p = p
        self.p_max = int(p_max)
        self.intercept = bool(intercept)

    def fit(self, x, y=None, **kwargs) -> TsVar:
        x = self._validate_data(x)
        self._set_metadata(x)
        data = pd.DataFrame(x)
        if self.target is None:
            self.target = self.y_name
        if self.target not in self.variables:
            raise ValueError("target must be one of the variables in ts_data_mv.")
        if self.p is None:
            best = _var_select_order(data, self.variables, self.p_max, intercept=self.intercept)
        else:
            best = _var_fit_order(data, self.variables, int(self.p), intercept=self.intercept)
        self.p = best["p"]
        self.model = best
        self.y_name = self.target
        self.x_names = [v for v in self.variables if v != self.y_name]
        return self

    def _next_row(self, history):
        arr = history[self.variables].to_numpy(dtype=float)
        values = [1.0] if self.model["intercept"] else []
        for lag in range(1, self.p + 1):
            values.extend(arr[arr.shape[0] - lag, :])
        return np.asarray(values) @ self.model["coefficients"]

    def predict(self, x=None, steps_ahead: int = 1, return_all: bool = False,
                **kwargs) -> TsMvPrediction:
        steps_ahead = int(steps_ahead)
        if x is None:
            history = pd.DataFrame(self.history)
        else:
            x = self._validate_data(x)
            history = pd.DataFrame(x)
        history = history[self.variables].copy()

        preds = np.zeros((steps_ahead, len(self.variables)))
        for i in range(steps_ahead):
            row = self._next_row(history)
            preds[i, :] = row
            history = pd.concat(
                [history, pd.DataFrame([row], columns=self.variables)], ignore_index=True)

        pred_df = pd.DataFrame(preds, columns=self.variables)
        self.history = adjust_ts_data_mv(history, y=self.y_name, x=self.x_names)
        prediction_y = pred_df[self.y_name].to_numpy()
        prediction_x = {name: pred_df[name].to_numpy() for name in self.x_names}
        return _mv_compose_prediction(self.y_name, self.x_names, self.variables,
                                      prediction_y, prediction_x)


# ===========================================================================
# ts_arimax.R -- ARIMA with external regressors
# ===========================================================================

def _fit_arimax(series, order, trend, exog):
    from statsmodels.tsa.arima.model import ARIMA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(series, order=order, trend=trend, exog=exog).fit()


def _auto_order_x(series, exog):
    """Compact AICc search approximating auto.arima with xreg."""
    best = None
    for d in (0, 1, 2):
        trend = "c" if d == 0 else ("t" if d == 1 else "n")
        for p in range(4):
            for q in range(4):
                if p == 0 and q == 0 and d == 0:
                    continue
                try:
                    res = _fit_arimax(series, (p, d, q), trend, exog)
                except Exception:  # noqa: BLE001, S112 - skip unfittable orders
                    continue
                if best is None or res.aicc < best[0]:
                    best = (res.aicc, (p, d, q), res)
    if best is None:
        return (1, 0, 0), _fit_arimax(series, (1, 0, 0), "c", exog)
    return best[1], best[2]


class TsArimax(TsRegMv):
    """ARIMA with external regressors (``ts_arimax``).

    The singular multivariate counterpart of :class:`~pytspredit.TsArima`: it
    forecasts the target with the aligned auxiliary variables entering as
    external regressors (``exog``). Unknown future auxiliaries are generated by
    the univariate models in ``models_x``.

    Parameters
    ----------
    models_x:
        Optional ``dict`` of one univariate model per auxiliary variable.
    p, d, q:
        Explicit ARIMA order; leave ``None`` for automatic selection.

    References:
        - Box GEP, Jenkins GM, Reinsel GC, Ljung GM (2015). Time Series Analysis:
          Forecasting and Control. Wiley.
        - Hyndman RJ, Athanasopoulos G (2021). Forecasting: Principles and
          Practice, 3rd ed. OTexts. https://otexts.com/fpp3/

    Porting:
        Mirrors ``ts_arimax.R``. Fidelity: approximate.
        - Wraps ``statsmodels`` ARIMA with ``exog`` where R used
          ``forecast::Arima``/``auto.arima`` with ``xreg``: explicit ``(p,d,q)``
          matches exactly; the automatic AICc search approximates ``auto.arima``
          (different MLE backend), so forecasts are close, not bit-identical.
    """

    def __init__(self, models_x=None, p=None, d=None, q=None) -> None:
        super().__init__(models_x=models_x)
        self.p, self.d, self.q = p, d, q
        self.manual_order = p is not None and d is not None and q is not None
        self.include_mean = bool(self.manual_order and d == 0)
        self.include_drift = bool(self.manual_order and d == 1)

    def fit(self, x, y=None, **kwargs) -> TsArimax:
        x = self._validate_data(x)
        self._set_metadata(x)
        self._validate_models_x()
        self._fit_aux_models(pd.DataFrame(x))
        data = pd.DataFrame(x)
        y_series = data[self.y_name].to_numpy(dtype=float)
        xreg = data[self.x_names].to_numpy(dtype=float)

        if self.manual_order:
            trend = _trend(self.d, self.include_mean, self.include_drift)
            self.model = _fit_arimax(y_series, (self.p, self.d, self.q), trend, xreg)
        else:
            (self.p, self.d, self.q), self.model = _auto_order_x(y_series, xreg)
            self.include_mean = self.d == 0
            self.include_drift = self.d == 1
        return self

    def predict(self, x=None, steps_ahead: int = 1, return_all: bool = False,
                **kwargs) -> TsMvPrediction:
        steps_ahead = int(steps_ahead)
        future_x = self._forecast_aux(x=x, steps_ahead=steps_ahead)
        exog = future_x[self.x_names].to_numpy(dtype=float)
        prediction_y = np.asarray(
            self.model.forecast(steps_ahead, exog=exog), dtype=float)
        self.history = self._update_history(future_x, prediction_y)
        prediction_x = {name: future_x[name].to_numpy(dtype=float) for name in self.x_names}
        return _mv_compose_prediction(self.y_name, self.x_names, self.variables,
                                      prediction_y, prediction_x)
