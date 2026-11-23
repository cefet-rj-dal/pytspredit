# Mapeamento R ↔ Python (tspredit ↔ pytspredit)

Correspondência entre a API pública do pacote R **tspredit** e o port Python
**pytspredit**. Convenções: **classes** R (construtor S3 em `snake_case`) viram
**PascalCase** no Python; **funções construtoras de dados** ficam em `snake_case`
(`ts_data`, `ts_projection`, `ts_sample` — idiomático, são funções como no R);
genéricos S3 (`fit`/`action`/`predict`/`transform`/`evaluate`/…) viram **métodos de
instância**; `data.frame` → `pandas.DataFrame`.

**Legenda:** ✅ portado (release) · 🔜 próxima release · ⬜ Ciclo 2 (dep pesada, extra) ·
❌ fora de escopo (ver §Fora de escopo).

## Núcleo de dados / contrato (Experiment Line)
| R (arquivo) | R (símbolo) | Python | Status |
|---|---|---|---|
| `ts_data.R` | `ts_data`, `ts_head`, `adjust_ts_data` | `ts_data`, `ts_head`, `adjust_ts_data` | ✅ 0.1.0 |
| `ts_projection.R` | `ts_projection` | `ts_projection`, `TsProjection` | ✅ 0.1.0 |
| `ts_sample.R` | `ts_sample` | `ts_sample`, `TsSample` | ✅ 0.1.0 |
| `ts_reg.R` | `ts_reg`, `do_fit`, `do_predict`, `MSE.ts`/`sMAPE.ts`/`R2.ts` | `TsReg`, `_do_fit`/`_do_predict` (hooks), `mse_ts`/`smape_ts`/`r2_ts` | ✅ 0.1.0 |
| `ts_regsw.R` | `ts_regsw` | `TsRegSW` (com `steps_ahead`) | ✅ 0.1.0 |
| `data.R` | `tsd` | `datasets.tsd()` | ✅ 0.1.0 |

## Normalização (`DalTransform`, com `inverse_transform`)
| R | Python | Status |
|---|---|---|
| `ts_norm_none.R` | `TsNormNone` | ✅ 0.1.0 |
| `ts_norm_gminmax.R` | `TsNormGminmax` | ✅ 0.1.0 |
| `ts_norm_swminmax.R` | `TsNormSwminmax` | ✅ 0.1.0 |
| `ts_norm_diff.R` | `TsNormDiff` | ✅ 0.1.0 |
| `ts_norm_an.R` + `tsanutils.R` (utils) | `TsNormAn` | ✅ 0.1.0 |

## Seleção de lag
| R | Python | Status |
|---|---|---|
| `ts_lagmap.R` + `tslagutils.R` (12 estratégias) | `TsLagmap` (12 estratégias) | ✅ 0.1.0 |

## Regressores
| R | Python | Status |
|---|---|---|
| `ts_knn.R` / `ts_mlp.R` / `ts_rf.R` / `ts_svm.R` | `TsKnn` / `TsMlp` / `TsRf` / `TsSvm` | ✅ 0.1.0 |
| `ts_elm.R` | `TsElm` (ELM próprio) | ✅ 0.1.0 |
| `ts_arima.R` | `TsArima` (statsmodels; estende `TsReg`) | ✅ 0.1.0 |
| `ts_deterministic.R` | `TsDeterministic`, `TsPeriodic`, `TsPersist` (nativos) | ✅ 0.2.0 |
| `ts_darima.R` | `TsDarima` (linear sobre janelas, `_RLinear`) | ✅ 0.2.0 |
| `ts_warma.R` | `TsWarma` (lm local + `ndiffs`) | ✅ 0.2.0 |

## Augmentation (`DalTransform` sobre a janela)
| R | Python | Status |
|---|---|---|
| `ts_aug_none.R` | `TsAugNone` | ✅ 0.1.0 |
| `ts_aug_flip/stretch/shrink/wormhole.R` | `TsAugFlip`/`TsAugStretch`/`TsAugShrink`/`TsAugWormhole` | ✅ 0.3.0 |
| `ts_aug_jitter/awareness/awaresmooth.R` | `TsAugJitter`/`TsAugAwareness`/`TsAugAwaresmooth` | ✅ 0.3.0 |

## Filtros (`ts_fil_*`, `DalTransform` sobre a série 1-D)
| R | Python | Status |
|---|---|---|
| `ts_fil_none/ma/ema/winsor/recursive/fft/smooth.R` | `TsFilNone`/`TsFilMa`/`TsFilEma`/`TsFilWinsor`/`TsFilRecursive`/`TsFilFft`/`TsFilSmooth` | ✅ 0.4.0 |
| `ts_fil_ses/qes/hp/lowess/seas_adj/kalman.R` | `TsFilSes`/`TsFilQes`/`TsFilHp`/`TsFilLowess`/`TsFilSeasAdj`/`TsFilKalman` | ✅ 0.4.0 |
| `ts_fil_spline/gabor.R` | `TsFilSpline`/`TsFilGabor` (scipy) | ✅ 0.4.0 |
| `ts_fil_emd.R` / `ts_fil_remd.R` | `TsFilEmd` / `TsFilRemd` | ⬜ 0.7.0 (extra `[emd]`, PyEMD) |
| `ts_fil_wavelet.R` | `TsFilWavelet` | ⬜ 0.8.0 (extra `[wavelet]`, PyWavelets) |

## Multivariado + plot
| R | Python | Status |
|---|---|---|
| `ts_data_mv.R` | `ts_data_mv`, `adjust_ts_data_mv`, `TsDataMv` | ✅ 0.5.0 |
| `ts_regsw_mv.R` | `TsRegswMv`, `ts_mv_spec`, `TsMvSpec`, `TsMvPrediction` | ✅ 0.5.0 |
| `ts_reg_mv.R` | `TsRegMv` | ✅ 0.5.0 |
| `ts_lm_mv.R` | `TsLmMv` (reusa `_RLinear`) | ✅ 0.5.0 |
| `ts_var.R` | `TsVar` (OLS nativo + AICc) | ✅ 0.5.0 |
| `ts_arimax.R` | `TsArimax` (statsmodels ARIMA + exog) | ✅ 0.5.0 |
| `ts_plot_mv.R` (`plot_ts_pred_mv`; `plot_ts_pred` vem do daltoolbox `grf_ts_pred.R`) | `plot_ts_pred`, `plot_ts_pred_mv` (matplotlib) | ✅ 0.5.0 |

## Tuning (sobre `DalTune`)
| R | Python | Status |
|---|---|---|
| `ts_tune.R` | `TsTune` | ✅ 0.1.0 (M4) |
| `ts_integtune.R` | `TsIntegtune` | ✅ 0.1.0 (M4) |

## Benchmark / datasets
| R | Python | Status |
|---|---|---|
| `tspredbench.R` (só doc de datasets; **sem harness no R**) | `TsPredBench` (harness datasets×modelos → DataFrame tidy) — interface nova, `Porting: n/a` | ✅ 0.6.0 |
| `data/*.RData` (miniaturas Eidolon com `attr(x,"url")`) | `datasets.<nome>()` (CSV/JSON + `manifest.json`); `catalog`/`list_datasets`/`dataset_url` | ✅ 0.6.0 |
| `tspredbench_loadfulldata.R` | `datasets.loadfulldata(nome)` (download remoto) | ⬜ 0.9.0 (extra `[bench]`) |

> **Gerador canônico das miniaturas:** `cefet-rj-dal/tspredbench` → `etl/02-build-minidata.R`
> (estendido) emite R `.RData` em `data/` **e** Python (CSV/JSON+manifest) em `data/python/`
> do mesmo objeto encolhido. Sincronizado para `src/pytspredit/data/` (+`tsd` local, sem url).

## Fora de escopo (❌)
- **Famílias de deep learning** que aparecem nos **notebooks** do R (`conv1d`, `lstm`,
  `torch-ts-mlp` em `prediction/` e `multivariate/`): **não têm fonte em `tspredit/R/`**
  (vêm de outro pacote da família, p.ex. `daltoolboxdp`/torch) e o pytspredit **não tem
  backend torch**. Lacuna deliberada — não serão portadas; os notebooks 1:1 correspondentes
  também não.

## Pendências de porte (dentro de escopo)
- **⬜ Ciclo 2:** `ts_fil_emd`/`ts_fil_remd` (0.7.0), `ts_fil_wavelet` (0.8.0),
  `tspredbench_loadfulldata.R` (0.9.0).

> Com o 0.6.0, o **Ciclo 1 está completo** — todo o `tspredit` portado exceto EMD/wavelet
> e o `loadfulldata` remoto (as únicas peças com dep pesada).

> **Auditoria dos 69 símbolos do `NAMESPACE`** (feita em 2026-11-19, no `planejamento.md`
> §5.4): todos os `export()` do R estão cobertos pelo roadmap — nenhum ficou de fora.
