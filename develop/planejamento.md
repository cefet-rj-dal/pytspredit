# Planejamento — Port do tspredit (R) para pytspredit (Python)

> Documento vivo. Atualizado a cada checkpoint. Mesmo método validado no
> **pydaltoolbox** (já publicado no PyPI). Objetivo: você saber todos os passos,
> o que já foi feito, o que vem, e por quê — sem surpresas.

Última atualização: 2026-11-18

---

## 1. O todo: o ecossistema DAL

```
          pydaltoolbox  (fundação: Linha de Experimento) — no PyPI
                 │
        ┌────────┴────────┐
   pytspredit         pyharbinger
 (previsão de TS)   (detecção de eventos em TS)
        (ESTE)        (depende dos dois)
```

`tspredit` depende **apenas** do `daltoolbox` → `pytspredit` depende apenas do
`pydaltoolbox` (já no PyPI). Este é o passo natural **antes** do `pyharbinger`.

Princípio central (herdado): **não reimplementar algoritmos** — envolver
`scikit-learn`/`numpy`/`pandas`/`statsmodels` com a interface uniforme da Linha de
Experimento (`fit`/`action`/`predict`/`transform`/`evaluate`/`set_params`). Nossa
contribuição é a **interface**, não os algoritmos.

---

## 2. Decisões travadas (2026-11-18)

| Tema | Decisão |
|---|---|
| Fundação | Reusar `pydaltoolbox` do PyPI (`Regression`, `Predictor`, `DalTransform`, `DalTune`, `adjust_*`). |
| Estilo de API | Métodos de instância; `fit` retorna `self` (encadeável). |
| Nomenclatura | **Classes** em PascalCase (`TsReg`, `TsKnn`, `TsNormGminmax`, `TsLagmap`); **funções construtoras de dados** em snake_case (`ts_data`, `ts_projection`, `ts_sample`) — idiomático (são funções, como no R). |
| Modelo de dados | Representação ts_data = `pandas.DataFrame` com colunas `t{sw-1}..t0` (plano, sem subclasse — facilita a aritmética das normalizações). |
| **Lag map** | **Portar as 12 estratégias na v1** (recent/even/geom/acf/pacf/peaks/seasonal/acf_seasonal/pacf_seasonal/blocks/mi/mrmr) — fidelidade total ao R. |
| **ARIMA/VAR** | **statsmodels = dependência core** da v1. `ts_arima`/`ts_arimax` (univariados) entram; `ts_var` é multivariado → vai com o bloco MV. |
| **Multivariado** | **Adiado para o Bloco B** (`ts_data_mv`, `ts_reg_mv`, `ts_regsw_mv`, `ts_lm_mv`, `ts_var`, `ts_plot_mv`). |
| Dependências v1 | `numpy`, `pandas`, `scikit-learn`, `statsmodels`, `pydaltoolbox`. |
| Fidelidade | Adapters fiéis aos defaults do backend R; **comparar resultado R vs Python** nos pontos sensíveis. |
| Publicação | Um único release (`0.1.0`) ao final da última fase. Eduardo faz/autoriza o upload (token `daldeploy` de conta inteira, já no `.pypirc`). PyPI sem prazo; versões imutáveis; beta até `1.0.0`. |
| Workflow | Branch por fase → push → Eduardo revisa no GitHub → ff-merge na `main`. `gh` não instalado. Verde sempre (ruff+pytest+build dos exemplos). |

---

## 3. Arquitetura a portar (árvore)

```
pydaltoolbox.Predictor
└── TsReg ───────── evaluate = MSE/sMAPE/R2 ; baseline predict = t0
    └── TsRegSW ─── preprocess (TsNorm*) + input_map (TsLagmap) + backend
        ├── TsKnn   → KNeighborsRegressor
        ├── TsMlp   → MLPRegressor
        ├── TsRf    → RandomForestRegressor
        ├── TsSvm   → SVR
        ├── TsElm   → ELM (implementação própria, leve)
        └── ...
    └── TsArima / TsArimax → statsmodels (operam na série, sem janela)

pydaltoolbox.DalTransform
├── TsNormNone / TsNormGminmax / TsNormSwminmax / TsNormDiff / TsNormAn
├── TsFil* (filtros)      — ma, ema, ses, qes, kalman, lowess, spline, winsor, hp...
└── TsAug* (augmentation) — jitter, flip, stretch, shrink, wormhole, awareness...

pydaltoolbox.DalBase
└── TsLagmap  — 12 estratégias de seleção de lag (fit aprende as posições)

pydaltoolbox.DalTune
└── TsTune / TsIntegtune — CV por janelas, minimiza erro
```

---

## 4. Estratégia de garantia (três camadas)

1. **Unitário** → `pytest` por componente (`fit → predict/transform → evaluate`).
2. **Integrado** → exemplos ponta-a-ponta (tutoriais).
3. **Documentação executável** → `notebooks/**/*.md` (jupytext) renderizado em
   `examples/**/*.md` (saída real embutida) por `tools/build_examples.py`. Falha se
   houver erro → é smoke test. Tabelas em texto puro (estilo `##` do R). **Um
   exemplo didático por classe** (objetivo + referência do artigo + fit/predict/evaluate).

**Comandos:** `pip install -e ".[dev,docs]"` · `ruff check .` · `pytest` ·
`python tools/build_examples.py`.

---

## 5. Roadmap

Legenda: ✅ feito · 🔜 próximo · ⬜ pendente (v1) · 🔭 Bloco B (pós-v1)

### BLOCO A — v1 publicável

#### ✅ Fase 0 — Scaffolding
`pyproject` (deps incl. `pydaltoolbox`+`statsmodels`), layout `src/pytspredit/`,
`tools/build_examples.py`, `develop/` (PROMPT+planejamento+deploy), README.

#### ✅ Fase 1 — Núcleo de dados
`ts_data` (janela deslizante), `ts_projection`, `ts_sample`, `adjust_ts_data`,
`ts_head`, `datasets.tsd()` (CSV extraído do `.RData` — idêntico ao R). Testes +
notebooks `tsdata/`. **Verificado R vs Python: dim/colunas/valores idênticos.**

> **Meta: publicar `0.1.0` HOJE.** Para caber em um dia, consolidei o roadmap em
> **milestones** (não mais 10 fases com 10 pausas). Resta decidir **o escopo de
> hoje** (ver §5.1) e a **cadência de revisão** (ver §5.2).

### 5.1 Escopo do 0.1.0 de hoje — ✅ DECIDIDO: Trilha A (Enxuta) + pausar a cada milestone (M2→M3→M4)

Duas trilhas. A recomendada é a **Enxuta**: entrega o *pipeline de previsão
univariado completo* (o contrato central e o que o pyharbinger consome), e deixa
filtros + augmentation como *fast-follow* `0.2.0` (poucos dias depois).

**TRILHA A — Enxuta (recomendada p/ hoje): 3 milestones restantes**

| # | Milestone | Conteúdo |
|---|---|---|
| ✅ M0 | Scaffolding | feito |
| ✅ M1 | Núcleo de dados | feito |
| ✅ **M2** | Substrato de modelagem | `TsReg` + `TsRegSW` (com `steps_ahead`) + **normalizações (5)** + `TsLagmap` (**12 estratégias**) — **paridade R vs Py verificada** (norms ≤1e-13, 12 lagmaps idênticos, evaluate idêntico) |
| ✅ **M3** | Regressores | `TsKnn`, `TsMlp`, `TsRf`, `TsSvm`, `TsElm` + `TsArima` (univariado) — **paridade R vs Py end-to-end verificada** (ver abaixo). *(`TsArimax` é multivariado → Bloco B.)* |
| ✅ **M4** | Tuning + fecho → 🚀 publica | `TsTune`, `TsIntegtune` (+ `TsAugNone` identidade) + README v1 + build/`twine check` OK + install em venv limpo. **Falta só o `twine upload` (Eduardo autoriza).** |

**Paridade M3 (tsd, previsão 5 passos, R vs Python):** knn `max|Δ|=4.7e-9` (exato) ·
elm `4.8e-9` (exato, equivale a OLS) · svm `2.5e-4` · arima `6.0e-4` · rf `3.6e-2` ·
mlp `2.2e-2`. Determinísticos (knn, elm) batem; svm/arima quase exatos; rf/mlp dentro
da tolerância de RNG do backend (esperado).

→ Filtros (13) e augmentation (8) viram **`0.2.0`** (uma milestone cada, dias depois).

**TRILHA B — Completa hoje: 4 milestones restantes**

Igual à Enxuta, mas insere antes do fecho:

| **M4** | Filtros + Augmentation | 13 filtros simples + 8 augmentations |
| **M5** | Tuning + fecho → 🚀 publica | idem M4 da trilha A |

> Risco da Trilha B: ~21 classes a mais com exemplo+teste+comparação R/Python no
> mesmo dia. Factível, mas aperta e sobra menos revisão sua. Por isso recomendo a A.

### 5.2 Cadência de revisão — DECISÃO EM ABERTO
- **(i) Pausar a cada milestone** (você revisa M2, M3, M4 no GitHub) — mais controle.
- **(ii) Correr direto até UMA revisão final** antes do `twine upload` — mais rápido;
  você aprova o pacote inteiro de uma vez. **Recomendada para fechar hoje.**

### 5.3 Convenção de exemplos (ajuste para caber hoje)
Mantemos **um exemplo por classe**, mas **agrupados por família num notebook** (ex.:
`normalization/` num só `.md` com uma seção por norma; idem regressores), em vez de
40 arquivos. Reduz o volume de docs sem perder o “didático por classe”.

## 5.4 Roadmap 0.2.0 → 1.0.0 (refinado) — cobertura de TODAS as classes R

> Objetivo: introduzir os componentes **gradualmente**, uma release = um tema
> coeso, **minimizando a entrada de dependências**. A 0.1.0 cobriu o pipeline de
> previsão univariado; daqui até a 1.0.0 cobrimos **todo o `tspredit`**.

### Política de dependências (o critério de decisão)
1. **Nativo (numpy/pandas)** quando o algoritmo é simples/estável → sem dep nova.
2. **Reutilizar deps já no core** (`scikit-learn`, `statsmodels`) quando cobrem
   fielmente o backend R → custo zero.
3. **Declarar dep leve e ubíqua** (`scipy`, já transitiva via sklearn/statsmodels)
   quando necessário → custo baixo; vira dep core.
4. **Deps pesadas/nicho → extras opcionais** (`pip install pytspredit[emd,wavelet,plot]`),
   introduzidas tarde e isoladas, para o core ficar enxuto.
5. Sempre **comparar R vs Python**; se a lib diverge do backend R, documentar a
   tolerância ou implementar nativo para fidelidade.

**Descobertas que barateiam o roadmap** (lendo os fontes R): `ts_darima` é um
regressor **linear sobre janelas** (não ARIMA distribuído), `ts_warma` é lm + passos
locais + `ndiffs`, `ts_deterministic` é regra pura, e o Kalman (`KFAS`) cabe em
`statsmodels` state space. Logo, as **únicas deps novas** até a 1.0.0 são `scipy`
e `matplotlib` (**core**, leves/ubíquas) e `PyEMD`/`PyWavelets`/leitor `rdata`
(**extras opcionais**).

### Releases — DOIS CICLOS (decidido com o Eduardo, 2026-11-19)

Mesma estrutura do `pyharbinger`: duas rodadas sobre os **mesmos temas**.
- **Ciclo 1 (baixa dependência):** varre *todos* os temas, mas só as classes **sem dep
  pesada nova** (nativo / sklearn / statsmodels / `scipy` / `matplotlib`). Entrega quase
  todo o `tspredit` com o core enxuto.
- **Ciclo 2 (completo):** volta aos *mesmos temas* acrescentando as classes de **maior
  dependência** — **uma família de dep por release**, cada uma como *extra* opcional.
  **Eduardo refina o Ciclo 2 em detalhe comigo.**

Granularidade livre (muitos ciclos pequenos OK; pode chegar a ~20 releases até a 1.0.0).
Cada release = branch por milestone, paridade R vs Py, testes + ruff + notebooks verdes,
links do GitHub, revisão do Eduardo, ff-merge.

#### Ciclo 1 — baixa dependência

- **✅ 0.2.0 — Zoo de modelos** *(0 dep nova)* — **PUBLICADO**: `TsDeterministic`
  (+`TsPeriodic`, `TsPersist`) nativo · `TsDarima` (lm sobre janelas) · `TsWarma`
  (lm + passos locais + `ndiffs` via statsmodels KPSS). **Paridade R vs Py (branch R
  corrigido): deterministic exato; darima/warma ≤4.8e-9** (diff e série logística
  bem-condicionadas). `_RLinear` replica o `lm` do R (descarta colunas colineares em
  vez de solução min-norm). **Tolerância documentada:** em design *exatamente*
  rank-deficiente (série com recorrência linear exata, p.ex. seno puro, com
  `input_size` acima do posto efetivo) R e Python escolhem soluções aliased diferentes
  — usar `TsNormDiff` ou `input_size` menor. **Achado:** confirmou que o bug do R
  `ts_norm_none` corrompe darima/warma (default preprocess); o patch está no branch
  `fix-ts-norm-none-target-path` do `tspredit` (o Python já está correto).
  **Correção da nota anterior:** NÃO há "tolerância de rank-deficiência". Contra o R
  **corrigido**, Python e R batem a ~1e-9 **inclusive** no caso degenerado (seno puro,
  posto 3) — ambos `[0.4121, 0.1739, -0.0752, -0.3195, -0.544]`. A divergência que eu
  havia visto (R=[0.87]) era 100% consequência do bug do `ts_norm_none` (alvo corrompido),
  não do ajuste linear. `_RLinear` reproduz o `lm` do R, descartando as mesmas colunas
  colineares. Ação pendente: mergear o patch do R para o R-main também concordar.
- **✅ 0.3.0 — Augmentation** *(0 dep, nativo)* — **PUBLICADO**: `TsAugFlip`,
  `TsAugStretch`, `TsAugShrink`, `TsAugWormhole` (determinísticos) + `TsAugJitter`,
  `TsAugAwareness`, `TsAugAwaresmooth` (RNG). Ativa o slot `augment` do `TsIntegtune`.
  **Paridade R vs Py:** determinísticos exatos (≤5e-15; wormhole 0.0); escala de ruído
  do jitter idêntica (0.4395701). Tolerância: jitter/awareness/awaresmooth usam RNG
  (rnorm/rexp/sample) → valores diferem do R (algoritmo + escala fiéis); `random_state`
  para reprodutibilidade.
- **⬜ 0.4.0 — Filtros (baixa dep)** *(+`scipy` core)*: `TsFilMa/Ema/None/Winsor/
  Recursive/Fft/Smooth` (nativo) · `TsFilSes/Qes/Hp/Lowess/SeasAdj/Kalman` (statsmodels)
  · `TsFilSpline/Gabor` (scipy). Sub-blocos 4a/4b se ajudar.
- **⬜ 0.5.0 — Multivariado + plot** *(+`matplotlib` core)*: `TsDataMv` +
  `adjust_ts_data_mv`, `TsRegMv`, `TsRegswMv` + `ts_mv_spec` (spec `model/variables/
  lags/transforms`), `TsLmMv` (sklearn) · `TsVar` (statsmodels VAR) · `TsArimax`
  (ARIMA+exog) · `plot_ts_pred_mv` (matplotlib).
- **⬜ 0.6.0 — Benchmark (parte leve)** *(0 dep nova)*: **miniaturas embarcadas**
  (CSV/JSON + `manifest.json` com urls, extraídas do `.RData` via `Rscript`) +
  `TsPredBench` (harness datasets×modelos). `datasets.<nome>()` devolve a miniatura.
  *(`loadfulldata` remoto → Ciclo 2.)*

> Fim do Ciclo 1: todo o `tspredit` portado **exceto** EMD/wavelet e o `loadfulldata`
> remoto (as únicas peças com dep pesada).

#### Ciclo 2 — dependências pesadas (extras opcionais) — *esboço, Eduardo refina*

Uma família de dep por release; cada classe faz import preguiçoso + erro acionável
nomeando o extra. Ordem/escopo a combinar com o Eduardo:
- **⬜ 0.7.0 — Filtros de decomposição (EMD):** `TsFilEmd`, `TsFilRemd` → extra `[emd]` (PyEMD).
- **⬜ 0.8.0 — Wavelet:** `TsFilWavelet` → extra `[wavelet]` (PyWavelets).
- **⬜ 0.9.0 — Datasets completos:** `datasets.loadfulldata(nome)` → extra `[bench]`
  (leitor RData puro-Python `rdata` + download das MESMAS urls remotas do repo
  `tspredbench`, como o R). Ver a nota do padrão miniatura+`loadfulldata` acima.

#### Fechamento
- **⬜ 1.0.0 — Paridade + estabilização:** varredura completa R vs Py, um exemplo por
  classe, matriz de paridade R↔Py, revisão/congelamento de API, docs. Sai de **beta**.

### Cobertura — auditoria dos 69 símbolos do `NAMESPACE` (2026-11-19)
Cruzamento completo `export()` do R × roadmap. **Todos cobertos:**
- **Já na 0.1.0:** `ts_data`, `ts_projection`, `ts_sample`, `ts_head`, `adjust_ts_data`,
  `MSE.ts`/`sMAPE.ts`/`R2.ts` (→ `mse_ts`/`smape_ts`/`r2_ts`), `ts_reg`, `ts_regsw`,
  `ts_lagmap` (+`tslagutils`), `ts_norm_none/gminmax/swminmax/diff/an` (+`tsanutils`),
  `ts_knn/elm/mlp/rf/svm`, `ts_arima`, `ts_tune`, `ts_integtune`, `ts_aug_none`.
- **`do_fit`/`do_predict`** (genéricos de extensão do R) → já cobertos como os hooks
  `_do_fit`/`_do_predict` do `TsRegSW` (0.1.0).
- **0.2.0:** `ts_deterministic`, `ts_periodic`, `ts_persist`, `ts_darima`, `ts_warma`.
- **0.3.0:** `ts_aug_jitter/flip/stretch/shrink/wormhole/awareness/awaresmooth`.
- **0.4.0:** `ts_fil_ma/ema/none/winsor/recursive/fft/smooth/ses/qes/hp/lowess/seas_adj/
  kalman/spline/gabor`.
- **0.5.0:** `ts_data_mv`, `adjust_ts_data_mv`, `ts_reg_mv`, `ts_regsw_mv`, `ts_mv_spec`,
  `ts_lm_mv`, `ts_var`, `ts_arimax`, `plot_ts_pred_mv`.
- **0.6.0:** datasets (miniaturas) + harness `TsPredBench`.
- **Ciclo 2:** `ts_fil_emd`/`ts_fil_remd` (0.7.0), `ts_fil_wavelet` (0.8.0),
  `loadfulldata` (0.9.0).

**Nenhum símbolo exportado ficou de fora.** (Correção desta auditoria: faltavam
`ts_mv_spec`, `adjust_ts_data_mv` e o nome real `plot_ts_pred_mv` — agora na 0.5.0.)

### Tabela de decisão por componente (dep)
| Grupo | Backend R | Decisão Python | Dep nova |
|---|---|---|---|
| augmentation (7) | nativo | numpy nativo | — |
| deterministic/periodic/persist | nativo | nativo | — |
| darima | `stats::lm` | sklearn/numpy lstsq | — |
| warma | `lm` + `ndiffs` | lm + ndiffs (statsmodels) | — |
| ma/ema/none/winsor/recursive/fft/smooth | nativo | numpy nativo | — |
| ses/qes/hp/lowess/seas_adj | forecast/stats | statsmodels | — |
| kalman | `KFAS` | statsmodels state space | — |
| spline/gabor | smooth.spline/— | scipy | **scipy** (core, leve) |
| lm_mv/var/arimax | —/—/forecast+xreg | sklearn / statsmodels | — |
| emd/remd | `hht` | PyEMD | **extra `[emd]`** ✅ |
| wavelet | `wavelets` | PyWavelets | **extra `[wavelet]`** ✅ |
| plot_mv | graphics | matplotlib | **matplotlib (core, leve)** — alinhado ao pyharbinger |
| datasets (miniatura) | `.RData` em `data/` | CSV/JSON no pacote | — ✅ (Ciclo 1) |
| `loadfulldata` | download `.RData` | leitor `rdata` + download | **extra `[bench]`** ✅ (Ciclo 2) |

---

## 6. Como trabalhamos
- Incremental, um passo por vez; você confere o `examples/` processado antes de escalar.
- Verde sempre (`ruff` + `pytest` + build dos exemplos).
- Decisões sempre registradas aqui.
- **Branch por milestone** (como no pydaltoolbox); **exemplos sempre commitados** e
  empurrados; eu te passo os **links do GitHub** de cada exemplo para você conferir.
  O PR você abre pelo link (sem `gh` aqui). Sem merge/commit na `main` sem você pedir.
- **Comparação R vs Python a cada passo (obrigatório):** para cada componente
  portado, rodar o equivalente no `tspredit` (R 4.5.0 disponível) e no `pytspredit`
  sobre o mesmo dado (`tsd`) e **conferir que os resultados batem** (dim/colunas/
  valores em dados; métricas MSE/sMAPE/R2 em modelos). Divergência esperada só onde
  o backend difere (ex.: RNG do MLP/RF) — nesses casos, documentar a tolerância. Isso
  dá segurança a cada etapa. (Já feito no núcleo: `ts_data` idêntico ao R.)

## 7. Pendências / decisões em aberto

### Decisões do roadmap 0.2.0→1.0.0
- [x] **Estrutura em DOIS CICLOS** (como no pyharbinger). ✅ Eduardo (2026-11-19):
      Ciclo 1 = varrer todos os temas só com classes de **baixa dependência**; Ciclo 2 =
      voltar aos mesmos temas com as classes de **maior dependência** (extras, uma família
      por release). **Eduardo refina o Ciclo 2 comigo.** Granularidade livre (~20 releases OK).
- [x] **`matplotlib` no core** (leve/ubíqua) → `TsPlotMv` entra no Ciclo 1 (0.5.0).
      ✅ Eduardo (2026-11-19), alinhado à decisão (A) do pyharbinger.
- [x] **`scipy` como dep core na 0.4.0** (spline/gabor/signal). ✅ Eduardo OK (2026-11-19).
- [x] **EMD/wavelet = extras opcionais** (`[emd]` PyEMD, `[wavelet]` PyWavelets), não nativo.
      ✅ Eduardo OK (2026-11-19).
- [x] **Datasets = miniatura-no-pacote + `loadfulldata` remoto, compatível com o R.**
      ✅ Eduardo (2026-11-19): manter o padrão do R — miniatura embutida + `loadfulldata`
      baixando o `.RData` completo das MESMAS urls do repo `tspredbench`. Ver §5.4/0.7.0.
- [x] Ordem do Ciclo 1 definida: 0.2 modelos → 0.3 augmentation → 0.4 filtros →
      0.5 multivariado+plot → 0.6 benchmark (miniaturas). Ciclo 2 (0.7+) = deps pesadas.
- [ ] **Sub-decisão (Ciclo 2, datasets):** formato das miniaturas (CSV p/ data.frame,
      JSON p/ listas) e leitor RData (`rdata` puro-Python vs `pyreadr`). Resolver na 0.9.0.
- [ ] **Granularidade dos filtros (0.4.0):** sub-blocos 4a/4b numa release vs dividir —
      confirmar ao começar a 0.4.0.
- [ ] **Ciclo 2 a refinar com o Eduardo** (ordem/escopo de emd/wavelet/bench).

### Resolvidas
- [x] Backend do ELM (M3): implementação própria (random hidden uniform_positive + pinv,
      sem bias). Com `actfun="purelin"` e `nhid >= nº de lags` equivale a OLS → bate com
      o R independente do RNG. Com `nhid` menor, projeção aleatória (diverge por RNG).
- [x] `TsArima` sobre statsmodels: ordem manual `(p,d,q)` fiel; `auto` via busca AICc
      (aproxima `forecast::auto.arima`, pode não reproduzir a ordem exata). Backend MLE
      difere do `forecast` → previsões próximas, não bit-idênticas.
- Nota: `TsArimax` é multivariado (xreg) → movido para o Bloco B junto da família MV.
- [x] **Fecho M4:** build + `twine check` OK; sdist exclui `develop/notebooks/examples/tests/tools`
      e inclui `tsd.csv`; **install em venv limpo OK** (numpy 2.5.5 / pandas 3.1.1 do PyPI).
      O teste de venv limpo pegou um bug real: `row.to_numpy()` vem **read-only** no numpy novo;
      corrigido com cópia explícita no `predict` multi-step do `TsRegSW`. 28 símbolos públicos.
      **Pronto para publicar — falta só o `twine upload` (Eduardo autoriza/roda).**
- [x] `ts_norm_an` portado e com paridade R vs Py (os 4 operadores). Nota: o default
      `outliers_boxplot` é aplicado sobre os dados já transformados (como no R) — crítico
      no `divide` (recorta explosões de divisão por média ~0).
- **Tolerância documentada (seasonality):** as 3 estratégias sazonais com `seasonality`
  explícito são idênticas ao R. Com `seasonality=None`, o R usa `forecast::findfrequency`
  (busca espectral AR); no Python uso uma estimativa por periodograma — o período pode
  divergir. Recomendação: passar `seasonality` quando quiser paridade exata.
- **Correção intencional (`TsNormNone`):** no caminho `x`, retorno `x` inalterado (o R
  retornava a matriz de janelas, o que quebraria o pipeline sliding-window).
