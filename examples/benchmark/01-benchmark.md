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

    21 datasets





    ['CATS',
     'EUNITE.Loads',
     'EUNITE.Reg',
     'EUNITE.Temp',
     'NN3',
     'NN5',
     'SantaFe.A',
     'SantaFe.D',
     'bioenergy',
     'climate',
     'emissions',
     'fertilizers',
     'gdp',
     'ipeadata.d',
     'ipeadata.m',
     'm1',
     'm3',
     'm4',
     'pesticides',
     'stocks',
     'tsd']




```python
# the contract for one dataset: kind, schema, and the remote URL of the full data
datasets.catalog()["NN5"]
```




    {'url': 'https://raw.githubusercontent.com/cefet-rj-dal/tspredbench/refs/heads/main/tspredit/NN5.RData',
     'kind': 'dataframe',
     'file': 'NN5.csv',
     'nrow': 30,
     'ncol': 112,
     'columns': ['NN5.001',
      'NN5.002',
      'NN5.003',
      'NN5.004',
      'NN5.005',
      'NN5.006',
      'NN5.007',
      'NN5.008',
      'NN5.009',
      'NN5.010',
      'NN5.011',
      'NN5.012',
      'NN5.013',
      'NN5.014',
      'NN5.015',
      'NN5.016',
      'NN5.017',
      'NN5.018',
      'NN5.019',
      'NN5.020',
      'NN5.021',
      'NN5.022',
      'NN5.023',
      'NN5.024',
      'NN5.025',
      'NN5.026',
      'NN5.027',
      'NN5.028',
      'NN5.029',
      'NN5.030',
      'NN5.031',
      'NN5.032',
      'NN5.033',
      'NN5.034',
      'NN5.035',
      'NN5.036',
      'NN5.037',
      'NN5.038',
      'NN5.039',
      'NN5.040',
      'NN5.041',
      'NN5.042',
      'NN5.043',
      'NN5.044',
      'NN5.045',
      'NN5.046',
      'NN5.047',
      'NN5.048',
      'NN5.049',
      'NN5.050',
      'NN5.051',
      'NN5.052',
      'NN5.053',
      'NN5.054',
      'NN5.055',
      'NN5.056',
      'NN5.057',
      'NN5.058',
      'NN5.059',
      'NN5.060',
      'NN5.061',
      'NN5.062',
      'NN5.063',
      'NN5.064',
      'NN5.065',
      'NN5.066',
      'NN5.067',
      'NN5.068',
      'NN5.069',
      'NN5.070',
      'NN5.071',
      'NN5.072',
      'NN5.073',
      'NN5.074',
      'NN5.075',
      'NN5.076',
      'NN5.077',
      'NN5.078',
      'NN5.079',
      'NN5.080',
      'NN5.081',
      'NN5.082',
      'NN5.083',
      'NN5.084',
      'NN5.085',
      'NN5.086',
      'NN5.087',
      'NN5.088',
      'NN5.089',
      'NN5.090',
      'NN5.091',
      'NN5.092',
      'NN5.093',
      'NN5.094',
      'NN5.095',
      'NN5.096',
      'NN5.097',
      'NN5.098',
      'NN5.099',
      'NN5.100',
      'NN5.101',
      'NN5.102',
      'NN5.103',
      'NN5.104',
      'NN5.105',
      'NN5.106',
      'NN5.107',
      'NN5.108',
      'NN5.109',
      'NN5.110',
      'NN5.111',
      'split']}



## Loading miniatures

Each accessor returns a natural Python structure for its kind.


```python
# data frame (wide format): one column per series / interval
datasets.santafe_a().head()
```




        V1  split
    0   86  train
    1  141  train
    2   95  train
    3   41  train
    4   22  train




```python
# list of series (dict of pandas Series)
bio = datasets.bioenergy()
{k: v.head(3).tolist() for k, v in bio.items()}
```




    {'usa_bio_consumption': [980185, 1130112, 1384457]}




```python
# list of data frames (dict); e.g. one ticker of the stocks collection
datasets.stocks()["VALE3"].head()
```




             date      open      high       low     close   volume
    0  2000-01-03  3.500000  3.542500  3.500000  3.500000   585600
    1  2000-01-04  3.466666  3.474166  3.416666  3.416666   782400
    2  2000-01-05  3.375000  3.416666  3.375000  3.416666  1876800
    3  2000-01-06  3.416666  3.500000  3.416666  3.416666   792000
    4  2000-01-07  3.458333  3.559166  3.458333  3.541666  5347200




```python
# nested list (dict of dict of series): competition series keyed by frequency
m4 = datasets.m4()
freq = list(m4)[0]
series = list(m4[freq])[0]
print("frequency:", freq, "| series:", series, "| length:", len(m4[freq][series]))
```

    frequency: hourly | series: H1 | length: 1008


## Resolving the full data (on demand)

The miniature carries the URL of the full `.RData` in the `cefet-rj-dal/tspredbench`
repository. Downloading and loading it (`loadfulldata`) ships in the optional
`[bench]` extra (a later release); the URL is already available here.


```python
datasets.dataset_url("NN5")
```




    'https://raw.githubusercontent.com/cefet-rj-dal/tspredbench/refs/heads/main/tspredit/NN5.RData'



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




         dataset    model           mse         smape        r2
    0        tsd  persist  5.991676e-01  1.507829e+00 -4.175042
    1        tsd      knn  9.658615e-04  1.575812e-01  0.991658
    2        tsd      mlp  6.998956e-04  1.471204e-01  0.993955
    3        tsd   darima  3.348299e-29  2.006428e-14  1.000000
    4  santafe_a  persist  1.846200e+03  8.066996e-01 -1.526826
    5  santafe_a      knn  1.032000e+02  1.604098e-01  0.858754
    6  santafe_a      mlp  3.310511e+01  1.087191e-01  0.954690
    7  santafe_a   darima  3.323907e+02  6.854741e-01  0.545069



> **What to observe.** The tidy table has one row per (dataset, model) pair. The
> naive `persist` baseline is the weakest; the linear `darima` with differencing is
> near-exact on the smooth `tsd` sine. A model that fails on a dataset yields `nan`
> metrics and an `error` column rather than aborting the whole grid.
