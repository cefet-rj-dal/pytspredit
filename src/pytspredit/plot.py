"""Forecast plotting (matplotlib).

Python counterpart of ``daltoolbox``'s ``plot_ts_pred`` (``grf_ts_pred.R``) and
of ``tspredit``'s ``ts_plot_mv.R`` (``plot_ts_pred_mv``).

``pydaltoolbox`` does not ship the ggplot2-based ``plot_ts_pred``, so this module
provides a small matplotlib equivalent: the observed series as a solid line with
points, the in-sample adjusted values as a dashed segment, and the out-of-sample
prediction as a second dashed segment. :func:`plot_ts_pred_mv` reuses it variable
by variable for the multivariate workflow.

References:
    - Hyndman RJ, Athanasopoulos G (2021). Forecasting: Principles and Practice,
      3rd ed. OTexts. https://otexts.com/fpp3/ (forecast visualization).

Porting:
    Mirrors ``grf_ts_pred.R`` (daltoolbox) and ``ts_plot_mv.R`` (tspredit).
    Fidelity: adapted.
    - Returns matplotlib ``Figure`` objects (R returned ``ggplot2`` objects); the
      geometry (solid observed line + dashed adjust + dashed prediction, the
      prediction segment starting at the last training index) matches R.
"""
from __future__ import annotations

import numpy as np

__all__ = ["plot_ts_pred", "plot_ts_pred_mv"]


def plot_ts_pred(x=None, y=None, yadj=None, ypred=None, label_x="", label_y="",
                 color="black", color_adjust="blue", color_prediction="green",
                 ax=None):
    """Plot an observed series with in-sample adjustment and optional prediction.

    ``yadj`` defines the length of the training segment; ``ypred`` (if given) is
    drawn after it. Returns the matplotlib :class:`~matplotlib.figure.Figure`.
    """
    import matplotlib.pyplot as plt

    y = np.asarray(y, dtype=float).ravel()
    if x is None:
        x = np.arange(1, len(y) + 1)
    x = np.asarray(x).ravel()
    yadj = np.asarray(yadj, dtype=float).ravel()
    ntrain = len(yadj)
    yhat = yadj
    ntest = 0
    if ypred is not None:
        ypred = np.asarray(ypred, dtype=float).ravel()
        yhat = np.concatenate([yhat, ypred])
        ntest = len(ypred)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
    else:
        fig = ax.figure

    ax.plot(x, y, color=color, marker="o", markersize=3, linewidth=1, label="observed")
    ax.plot(x[:ntrain], yhat[:ntrain], color=color_adjust, linestyle="--", label="adjust")
    if ntest > 0:
        lo = ntrain - 1
        hi = ntrain + ntest
        ax.plot(x[lo:hi], yhat[lo:hi], color=color_prediction, linestyle="--",
                label="prediction")

    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def plot_ts_pred_mv(history, future=None, prediction=None, variable=None,
                    label_x="", label_y="Value", color="black",
                    color_adjust="blue", color_prediction="green"):
    """Plot observed and forecast paths for the multivariate workflow.

    Reuses :func:`plot_ts_pred` for every variable in ``prediction`` (a
    :class:`~pytspredit.TsMvPrediction` returned by a multivariate ``predict``).
    Returns a single :class:`~matplotlib.figure.Figure` when ``variable`` is
    given, otherwise a ``dict`` of figures keyed by variable name.

    Parameters
    ----------
    history:
        Observed history (``ts_data_mv`` or DataFrame) used as context.
    future:
        Optional held-out aligned observations, shown with the predictions.
    prediction:
        Multivariate forecast returned by ``predict`` in the MV workflows.
    variable:
        Optional single variable to plot (default: all variables).
    """
    import pandas as pd

    variables = getattr(prediction, "variables", None)
    y_name = getattr(prediction, "y_name", None)
    x_names = getattr(prediction, "x_names", None)
    system = getattr(prediction, "system", None)
    if variables is None or y_name is None or x_names is None:
        raise ValueError("prediction must be a multivariate prediction returned by predict().")

    history = pd.DataFrame(history)
    future = None if future is None else pd.DataFrame(future)

    if system is not None:
        prediction_values = {name: np.asarray(system[name], dtype=float)
                             for name in [y_name, *x_names]}
    else:
        prediction_x = getattr(prediction, "prediction_x", {})
        prediction_values = {y_name: np.asarray(prediction, dtype=float)}
        prediction_values.update({k: np.asarray(v, dtype=float)
                                  for k, v in prediction_x.items()})

    if variable is not None:
        if variable not in variables:
            raise ValueError("Requested variable is not present in prediction.")
        variables = [variable]

    plots = {}
    single = len(variables) == 1
    for name in variables:
        observed_history = history[name].to_numpy(dtype=float)
        observed_future = (np.empty(0) if future is None
                           else future[name].to_numpy(dtype=float))
        observed = np.concatenate([observed_history, observed_future])
        plots[name] = plot_ts_pred(
            x=np.arange(1, len(observed) + 1),
            y=observed,
            yadj=observed_history,
            ypred=prediction_values[name],
            label_x=label_x,
            label_y=label_y if single else f"{label_y}: {name}",
            color=color, color_adjust=color_adjust, color_prediction=color_prediction)

    if variable is not None:
        return plots[variable]
    return plots
