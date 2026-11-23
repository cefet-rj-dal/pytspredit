---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Benchmark: dataset miniatures and the TsPredBench harness

This release ships the **Eidolon miniatures** of the `tspredbench` collection and a
small benchmarking harness, with **no new dependency**.

* **Miniatures** are compact, package-native stand-ins for a larger remote
  collection: the first 30 rows of a data frame (or the first element of a list),
  carrying the remote URL of the full artifact. The miniature and the full data
  share one usage contract, so local discovery and documentation work without
  downloading gigabytes. Values are extracted from the original R `.RData`
  miniatures, so they are numerically identical to the R package.
* **`TsPredBench`** runs a grid of *datasets x models* over the canonical
  Experiment Line path and returns a tidy table of MSE / sMAPE / R2.

**Reference:** Calmon, Junior, Santos, Alexandrino, Coutinho, Guedes, Carvalho &
Ogasawara (2026), *Eidolon: An Architecture for the Reproducible Publication of
Scientific Data Collections*, XX Brazilian e-Science Workshop (BreSci); Ogasawara
et al. (2025), *A Benchmark Time Series Dataset Collection for Prediction Models*.

```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts
from pytspredit import datasets
```

## The dataset catalog

`datasets.list_datasets()` lists every available miniature; `catalog()` returns the
manifest (URL, kind, schema) and `dataset_url(name)` resolves the full artifact.

```python
print(len(datasets.list_datasets()), "datasets")
datasets.list_datasets()
```

```python
# the contract for one dataset: kind, schema, and the remote URL of the full data
datasets.catalog()["NN5"]
```

## Loading miniatures

Each accessor returns a natural Python structure for its kind.

```python
# data frame (wide format): one column per series / interval
datasets.santafe_a().head()
```

```python
# list of series (dict of pandas Series)
bio = datasets.bioenergy()
{k: v.head(3).tolist() for k, v in bio.items()}
```

```python
# list of data frames (dict); e.g. one ticker of the stocks collection
datasets.stocks()["VALE3"].head()
```

```python
# nested list (dict of dict of series): competition series keyed by frequency
m4 = datasets.m4()
freq = list(m4)[0]
series = list(m4[freq])[0]
print("frequency:", freq, "| series:", series, "| length:", len(m4[freq][series]))
```

## Resolving the full data (on demand)

The miniature carries the URL of the full `.RData` in the `cefet-rj-dal/tspredbench`
repository. Downloading and loading it (`loadfulldata`) ships in the optional
`[bench]` extra (a later release); the URL is already available here.

```python
datasets.dataset_url("NN5")
```

## Running a benchmark grid

Pass a mapping of named univariate series and a mapping of named model factories.
Each model is fit on the training windows and evaluated on a multi-step forecast of
the held-out tail.

```python
bench = ts.TsPredBench(
    datasets={
        "tsd": datasets.tsd()["y"],
        "santafe_a": datasets.santafe_a()["V1"],
    },
    models={
        "persist": lambda: ts.TsPersist(),
        "knn": lambda: ts.TsKnn(ts.TsNormGminmax(), input_size=5),
        "mlp": lambda: ts.TsMlp(ts.TsNormGminmax(), input_size=5),
        "darima": lambda: ts.TsDarima(ts.TsNormDiff(), input_size=5),
    },
    input_size=5, steps_ahead=5,
)
bench.run()
```

> **What to observe.** The tidy table has one row per (dataset, model) pair. The
> naive `persist` baseline is the weakest; the linear `darima` with differencing is
> near-exact on the smooth `tsd` sine. A model that fails on a dataset yields `nan`
> metrics and an `error` column rather than aborting the whole grid.
