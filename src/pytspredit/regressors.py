"""Machine-learning sliding-window regressors.

Python counterpart of ``ts_knn.R``, ``ts_elm.R``, ``ts_rf.R``, ``ts_svm.R`` and
``ts_mlp.R``. Each is a thin :class:`~pytspredit.TsRegSW` subclass wrapping a
scikit-learn backend (ELM is implemented directly), reproducing the R backend's
default behaviour.

Determinism vs R:

* ``TsKnn`` (FNN ``knn.reg`` -> ``KNeighborsRegressor``) is deterministic and
  matches R exactly.
* ``TsElm`` with ``actfun="purelin"`` and ``nhid >= n_lags`` reduces to ordinary
  least squares, so it matches R regardless of the random hidden weights; with a
  smaller ``nhid`` the random projection makes it backend-RNG dependent.
* ``TsRf`` and ``TsMlp`` depend on the backend RNG, so results are close but not
  bit-identical to ``randomForest`` / ``nnet`` (expected tolerance).
* ``TsSvm`` wraps scaling to mirror ``e1071`` (``scale=TRUE``); it is deterministic
  and close to R up to the scaling/solver details.
"""
from __future__ import annotations

from typing import ClassVar

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .ts_reg import TsRegSW

__all__ = ["TsElm", "TsKnn", "TsMlp", "TsRf", "TsSvm"]


def _X(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _y(y) -> np.ndarray:
    return np.asarray(y, dtype=float).ravel()


class TsKnn(TsRegSW):
    """k-Nearest-Neighbors regression (``ts_knn``; FNN ``knn.reg``)."""

    def __init__(self, preprocess=None, input_size=None, input_map=None, k=None) -> None:
        super().__init__(preprocess, input_size, input_map)
        self.k = k

    def _do_fit(self, x, y):
        k = self.k if self.k is not None else max(1, int(self.input_size / 3))
        self.model = KNeighborsRegressor(n_neighbors=int(k))
        self.model.fit(_X(x), _y(y))
        return self

    def _do_predict(self, x):
        return self.model.predict(_X(x))


class TsRf(TsRegSW):
    """Random Forest regression (``ts_rf``; R ``randomForest``)."""

    def __init__(self, preprocess=None, input_size=None, input_map=None,
                 nodesize=1, ntree=100, mtry=None, random_state=0) -> None:
        super().__init__(preprocess, input_size, input_map)
        self.nodesize = nodesize
        self.ntree = ntree
        self.mtry = mtry
        self.random_state = random_state

    def _do_fit(self, x, y):
        x = _X(x)
        mtry = self.mtry if self.mtry is not None else int(np.ceil(x.shape[1] / 3))
        self.model = RandomForestRegressor(
            n_estimators=self.ntree,
            min_samples_leaf=self.nodesize,
            max_features=min(mtry, x.shape[1]),
            random_state=self.random_state,
        )
        self.model.fit(x, _y(y))
        return self

    def _do_predict(self, x):
        return self.model.predict(_X(x))


class TsSvm(TsRegSW):
    """Support Vector Regression (``ts_svm``; e1071 ``svm``, epsilon-SVR).

    Mirrors ``e1071``'s ``scale=TRUE`` default by standardizing inputs and target.
    """

    def __init__(self, preprocess=None, input_size=None, input_map=None,
                 kernel="radial", epsilon=0.0, cost=10.0) -> None:
        super().__init__(preprocess, input_size, input_map)
        self.kernel = kernel
        self.epsilon = epsilon
        self.cost = cost

    _KERNELS: ClassVar[dict] = {
        "radial": "rbf", "linear": "linear", "polynomial": "poly", "sigmoid": "sigmoid",
    }

    def _do_fit(self, x, y):
        svr = SVR(kernel=self._KERNELS[self.kernel], C=self.cost,
                  epsilon=self.epsilon, gamma="scale")
        self.model = TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), svr),
            transformer=StandardScaler(),
        )
        self.model.fit(_X(x), _y(y))
        return self

    def _do_predict(self, x):
        return self.model.predict(_X(x))


class TsMlp(TsRegSW):
    """Multilayer Perceptron regression (``ts_mlp``; R ``nnet``)."""

    def __init__(self, preprocess=None, input_size=None, input_map=None,
                 size=None, decay=0.01, maxit=1000, random_state=0) -> None:
        super().__init__(preprocess, input_size, input_map)
        self.size = size
        self.decay = decay
        self.maxit = maxit
        self.random_state = random_state

    def _do_fit(self, x, y):
        size = self.size if self.size is not None else int(np.ceil(self.input_size / 3))
        self.model = MLPRegressor(
            hidden_layer_sizes=(int(size),),
            activation="logistic",      # nnet uses a logistic hidden layer
            solver="lbfgs",             # nnet optimizes with BFGS
            alpha=self.decay,           # nnet 'decay' is L2 weight decay
            max_iter=self.maxit,
            random_state=self.random_state,
        )
        self.model.fit(_X(x), _y(y))
        return self

    def _do_predict(self, x):
        return self.model.predict(_X(x))


def _elm_activation(name):
    funcs = {
        "purelin": lambda z: z,
        "sig": lambda z: 1.0 / (1.0 + np.exp(-z)),
        "relu": lambda z: np.maximum(0.0, z),
        "radbas": lambda z: np.exp(-(z ** 2)),
        "tribas": lambda z: np.maximum(0.0, 1.0 - np.abs(z)),
    }
    if name not in funcs:
        raise ValueError(f"unknown ELM actfun '{name}'")
    return funcs[name]


class TsElm(TsRegSW):
    """Extreme Learning Machine regression (``ts_elm``; elmNNRcpp).

    Random hidden weights + closed-form (pseudo-inverse) output weights, no bias
    (matching ``init_weights='uniform_positive'``, ``bias=FALSE``).
    """

    def __init__(self, preprocess=None, input_size=None, input_map=None,
                 nhid=None, actfun="purelin", random_state=0) -> None:
        super().__init__(preprocess, input_size, input_map)
        self.nhid = nhid
        self.actfun = actfun
        self.random_state = random_state

    def _do_fit(self, x, y):
        x = _X(x)
        nhid = self.nhid if self.nhid is not None else max(1, int(self.input_size / 3))
        rng = np.random.default_rng(self.random_state)
        self._W = rng.uniform(0.0, 1.0, size=(x.shape[1], int(nhid)))  # uniform_positive
        self._act = _elm_activation(self.actfun)
        h = self._act(x @ self._W)
        self._beta = np.linalg.pinv(h) @ _y(y)
        self.model = self  # marker; prediction uses stored weights
        return self

    def _do_predict(self, x):
        h = self._act(_X(x) @ self._W)
        return h @ self._beta
