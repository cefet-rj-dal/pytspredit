# Planejamento — Port do tspredit (R) para pytspredit (Python)

> Documento vivo. Atualizado a cada checkpoint. Mesmo método validado no
> **pydaltoolbox** (já publicado no PyPI). Objetivo: você saber todos os passos,
> o que já foi feito, o que vem, e por quê — sem surpresas.

Última atualização: 2026-11-23

---

## Convenção de documentação (Eduardo, 2026-11-21)

**Toda classe Python deve trazer, no próprio docstring, (1) como o algoritmo
funciona + base teórica e (2) as referências acadêmicas — espelhando os
`@details` e `@references` do roxygen do pacote R.** Não basta descrever o
comportamento ou as escolhas fiéis ao R; incluir uma seção `References:` com as
citações copiadas do arquivo R correspondente. Vale para classes novas e para
preencher as já existentes. (Mesma regra adotada no `pydaltoolbox`.)

Além disso, **cada classe deve ter uma seção `Porting:` consistente e auditável**,
registrando a fidelidade ao R:

```
Porting:
    Mirrors ``<arquivo_r>.R``. Fidelity: exact|approximate|adapted.
    - <cada adaptação/ajuste deliberado e o porquê>   # só quando approximate/adapted
```

**Apenas o nome do arquivo R** — sem número de versão e sem citar o pacote R (já se sabe
de onde vem). Níveis de fidelidade: **exact** = comportamento igual ao R; **approximate**
= mesmo algoritmo, mas diferenças numéricas menores (backend/solver/RNG diferente, ou
aproxima um utilitário do R); **adapted** = mudança deliberada de comportamento,
reimplementação nativa ou correção de bug do R. Consolidar nessa seção as notas de
"Mirrors/replicates/approximates R" que hoje ficam soltas na prosa.

> **✅ Varredura concluída na 0.5.0 (2026-11-21).** Todas as classes públicas agora
> trazem `References:` (citações copiadas do `@references` do arquivo R correspondente,
> com artefatos de encoding corrigidos) e `Porting:` (arquivo R + nível de fidelidade +
> adaptações). Cobertos: `ts_reg`, `tsdata`, `datasets`, `norm` (5), `lagmap`,
> `regressors` (5), `arima`, `deterministic` (3), `darima_warma` (2), `augment` (8),
> `filters` (15), `tune` (2) e `multivariate` (0.5.0). Fica o padrão para as próximas releases.

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

## 1b. Princípio de longo prazo — back-implementation (Python = releitura, R = backport)

> **Direção estratégica (2026-11-20, Eduardo), comum aos três pacotes DAL
> (`pydaltoolbox`, `pytspredit`, `pyharbinger`).**

A família DAL adota, ao longo do tempo, uma **inversão de papéis**:

- **A versão Python é a releitura** — reimplementação própria (nativa) dos
  trabalhos anteriores, **dona dos algoritmos**, não apenas um *wrapper* de libs
  de terceiros.
- **A versão R passa a ser um backport da versão Python** — o Python vira a
  referência canônica e o R acompanha (invertendo a direção histórica R→Python).

Isso deixa o trabalho **sólido e autocontido**: a paridade fica **sob nosso
controle**, não refém dos defaults de pacotes de terceiros.

**Trilha de back-implementation (gradual e seletiva):** trocar dependências de
terceiros pela nossa implementação quando a lib **falta / não é mantida / causa
atrito (wheel/compilação) / impede a paridade**. Libs canônicas e estáveis
(`scikit-learn`, `scipy`, `numpy`, `pandas`) **seguem embrulhadas** salvo atrito
real. Candidato principal aqui: a **auto-seleção do `statsmodels`/`forecast`**
(ARIMA/ETS) — quando a seleção automática divergir, uma seleção própria dá
controle do resultado.

**Ação 1 — gate "diferença pequena":** ao portar nativo, **medir o gap vs. o
pacote de referência** (R / lib original) nos exemplos; aceitar só se pequeno,
documentando a tolerância. Gap grande → investigar antes.

**Trade-off explícito:** isto flexiona o princípio "não reimplementar" acima —
reimplementamos nos casos citados e seguimos embrulhando as libs canônicas.

**⚠️ Não é regra cega — é planejamento de DISCUSSÃO (caso a caso).** A inversão
*Python = releitura / R = backport* é a **tendência**, não um absoluto, e **não se
aplica a tudo**. O alvo são **pacotes atípicos / de nicho** (sem equivalente mantido,
que causam atrito de build, ou cuja auto-seleção impede a paridade) — **não** as libs
canônicas e comuns. Quando já existe uma implementação **madura** de um dos lados e a
**diferença é pequena**, mantém-se o que há — ex.: o **MLP do `scikit-learn` não vira
back-implementation**, porque o R já tem um MLP nativo excelente e a diferença é
pequena. Cada candidato é **decidido em conjunto**, caso a caso, nunca por automação.

---

## 1c. Práticas de maturidade herdadas do pydaltoolbox (decidido 2026-11-23)

No `pydaltoolbox`, **depois** do port funcional completo, vieram quatro passadas de
maturidade (0.21→0.24) que o elevaram de "portado" para "auditável e publicável como
referência". Trazemos as **mesmas práticas** para o pytspredit. Documentos-fonte:
`C:\Git\dal\pydaltoolbox\develop\{historico,mapeamento,parity}.md` e o passo 0.24.0 do
histórico de lá. Decisões com o Eduardo (2026-11-23):

**(A) Três documentos de desenvolvimento — criados AGORA (scaffold):**
- `develop/historico.md` — relato **cronológico** do trabalho concluído (0.1.0→0.5.0),
  separado do roadmap. (Antes o histórico vivia só na memória + prosa deste arquivo.)
- `develop/mapeamento.md` — tabela de **correspondência R↔Python** símbolo a símbolo
  (como a do daltoolbox), com status por classe (✅ portado / 🔜 0.6.0 / ⬜ Ciclo 2 /
  ❌ fora de escopo).
- `develop/parity.md` — **metodologia de verificação de paridade** reprodutível
  (compartilha a metodologia do `parity.md` do pydaltoolbox, adaptada às famílias de
  séries temporais; serve de espinha dorsal para artigo).

**(B) Ritual de paridade COMMITTADA — a partir do 0.6.0:** até aqui a paridade R vs
Python foi feita *ad-hoc* (números anotados na memória/planejamento). Dali em diante,
cada release segue o protocolo do pydaltoolbox: **congelar o input em CSV**, rodar os
dois lados sobre as MESMAS linhas, comparar o **invariante da família**, e deixar o
resultado **travado em `tests/test_quality_*.py`** + o **script reprodutível em
`develop/parity/`**. (Retroação de 0.1→0.5 entra na passada final de paridade.)

**(C) Exemplos 1:1 por classe — passada grande perto da 1.0.0:** hoje há 12 notebooks
**agrupados por família**; o alvo é **um notebook por classe portada**, espelhando a
árvore e a didática do R (About / Didactic goal / setup / load / split / fit / eval /
What to observe / References). **Fora de escopo:** famílias de deep learning que aparecem
nos notebooks do R (`conv1d`/`lstm`/`torch-ts-mlp`) — **não** têm fonte em `tspredit/R/`
(vêm de outro pacote) e o pytspredit não tem backend torch; documentado como lacuna
deliberada no `mapeamento.md`. Os exemplos de `datasets/` dependem das miniaturas do
0.6.0, então fecham **depois** dele.

**(D) Auditoria de crédito (estilo 0.24.0) — passada grande perto da 1.0.0, Python + R:**
varredura de TODAS as docstrings: (1) **zero frases proibidas** ("native implementation"/
"not a wrapper"/"no external dependency"/alfinetadas); (2) toda classe de **algoritmo**
com Teoria + `References:` (paper original) + `Porting:`; (3) **atribuição canônica** das
back-implementations ("Canonical; o R é backport desta", paper original + descrição da
inovação ++), p.ex. `_RLinear`, ELM, `TsVar` nativo, filtros nativos; (4) corrigir
referências `Mirrors X.R` a arquivos inexistentes. **Espelhada na roxygen do `tspredit`
R** e os fixes de paridade vão para os **dois lados** (como no daltoolbox).

> **Sequência (decidida):** scaffold dos docs + ritual de paridade **agora/0.6.0**;
> exemplos 1:1 e auditoria de crédito como **passadas finais** perto da 1.0.0 (ordem do
> pydaltoolbox: fazer o port inteiro e só então maturar). Ver §5.4/Maturidade, que
> **espelha o pydaltoolbox 0.20→0.24** em 6 passos (M1 docs → M2 exemplos 1:1 → M3 testes
> pareados retroativos → M4 metadados/Beta → M5 parity.md → M6 auditoria de crédito).
> **O último passo é um release BETA `1.0.0b1`** (rodar ~1 semana), e só depois a `1.0.0`
> estável (Eduardo, 2026-11-23) — não ir direto à 1.0.0.

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
   `examples/**/*.md` (saída real embutida) por `notebooks/conversor.py`. Falha se
   houver erro → é smoke test. Tabelas em texto puro (estilo `##` do R). **Um
   exemplo didático por classe** (objetivo + referência do artigo + fit/predict/evaluate).

**Comandos:** `pip install -e ".[dev,docs]"` · `ruff check .` · `pytest` ·
`python notebooks/conversor.py`.

---

## 5. Roadmap

Legenda: ✅ feito · 🔜 próximo · ⬜ pendente (v1) · 🔭 Bloco B (pós-v1)

### BLOCO A — v1 publicável

#### ✅ Fase 0 — Scaffolding
`pyproject` (deps incl. `pydaltoolbox`+`statsmodels`), layout `src/pytspredit/`,
`notebooks/conversor.py`, `develop/` (PROMPT+planejamento+deploy), README.

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
- **✅ 0.4.0 — Filtros (baixa dep)** *(+`scipy` core)* — **PUBLICADO**: 16 filtros.
  **Paridade R vs Py:** 9 EXATOS (~5e-9): `none/ma/ema/winsor/recursive/fft/smooth`,
  `hp` (solve manual) e `gabor` (kernel+convolução manual); `lowess` (statsmodels +
  `delta=0.01*(n-1)`, it=3) e `kalman` (KF local-level com init difusa) também batem
  (~5e-9); `ses` quase exato (8e-5). **Tolerância de backend (documentada):** `qes`
  (HoltWinters-trend, ~0.13), `seas_adj` (STL R vs statsmodels, ~1.2), `spline`
  (`smooth.spline` vs `scipy.make_smoothing_spline`, ~0.14) — mesma natureza das
  tolerâncias de ARIMA/RF/MLP; parâmetros batem, backend difere.
- **✅ 0.5.0 — Multivariado + plot** *(+`matplotlib` core)* — **IMPLEMENTADO** (branch
  `milestone-0.5.0-multivariate`, aguardando revisão/`twine` do Eduardo): `multivariate.py`
  = `TsDataMv` (subclasse de `DataFrame` com metadados) + `ts_data_mv`/`adjust_ts_data_mv`
  + `ts_mv_spec`/`TsMvSpec` + `TsRegswMv` (orquestrador sliding-window) + `TsRegMv` (base
  singular aligned) + `TsLmMv` (reusa `_RLinear`=`lm` do R) + `TsVar` (OLS nativo + AICc) +
  `TsArimax` (statsmodels ARIMA+exog) + `TsMvPrediction` (ndarray com `.system`). `plot.py`
  = `plot_ts_pred` (port matplotlib do `grf_ts_pred` do daltoolbox, ausente no pydaltoolbox)
  + `plot_ts_pred_mv`. **Paridade R vs Py (orquestração exata):** `TsVar` ~1e-6 (exato,
  nativo) · `TsLmMv` (futuro conhecido) 4.97e-9 (exato, `_RLinear`=`lm`) · `TsRegswMv`
  (alvo KNN + aux periódico) **idêntico a 8 casas** · `TsArimax` tolerância de backend
  (~1.4e-1 com xreg, statsmodels vs `forecast::Arima`) · ELM/RF/MLP como alvo MV =
  tolerância RNG/backend (documentada). **Achado reusável:** ELM-purelin só é OLS *exato*
  (invariante a RNG) quando `nhid == nº de features` (W quadrado invertível); com
  `nhid > features` a extrapolação multi-step depende da projeção aleatória (em R e Py).
  **Dado dos exemplos R:** `x2 = stats::filter(y, rep(1/3,3), sides=1)` dá NA nos 2
  primeiros (→ `y`); no Python usar `rolling(3)` (min_periods=3) + preencher NA com `y`
  (não `rolling(min_periods=1)`). 140 testes, 68 símbolos, 12 notebooks verdes.
- **✅ 0.6.0 — Benchmark (parte leve)** *(0 dep nova)* — **IMPLEMENTADO** (branch
  `milestone-0.6.0-benchmark`, aguardando revisão/`twine` do Eduardo): **miniaturas
  Eidolon embarcadas** (CSV p/ data.frame, JSON p/ list, + `manifest.json` com
  `{url, kind, schema}`) — 20 datasets + `tsd` via `datasets.<nome>()`
  (`cats/eunite_loads/eunite_reg/eunite_temp/nn3/nn5/santafe_a/santafe_d/ipeadata_d/
  ipeadata_m/bioenergy/climate/emissions/fertilizers/gdp/pesticides/m1/m3/m4/stocks`)
  + `catalog()`/`list_datasets()`/`dataset_url()` · `TsPredBench` (harness
  datasets×modelos → DataFrame tidy `[dataset, model, mse, smape, r2]`, factories ou
  instâncias, célula que falha vira `nan`+`error` sem abortar a grade). *(`loadfulldata`
  remoto → Ciclo 2 / 0.9.0.)* **Arquitetura Eidolon** (Calmon et al., 2026): miniatura
  compacta + url do artefato completo, preservando o contrato (identidade/schema/url);
  a serialização Python (CSV/JSON+manifest) é a instanciação do contrato, não do
  `.RData`. **Gerador canônico dual** no `tspredbench`: `etl/02-build-minidata.R`
  estendido para emitir, do MESMO objeto encolhido, o R `.RData` em `data/` (como hoje)
  **e** as miniaturas Python em `data/python/` — R↔Python idênticos por construção;
  sincronizadas para `src/pytspredit/data/` (+ `tsd` local). **Primeira release sob o
  ritual de paridade committada** (§1c-B): `develop/parity/datasets.R` gera
  `datasets_ref.json` (dos `.RData` mini) e `tests/test_quality_datasets.py` compara as
  miniaturas embarcadas ao R elemento-a-elemento (≤1e-9, R-free no teste). **Paridade:
  as 20 miniaturas batem exatamente com o R** (numérico + colunas de texto `split`/`date`;
  `_RLinear`-style N/A aqui). Dois achados reusáveis: (1) `rdata` (puro-Python) descarta
  `attr(url)` e puxa `xarray` → CSV/JSON+manifest é mais leve e fiel ao contrato; (2)
  colunas totalmente nulas do `jsonlite` viram `object` no pandas (vs `NA_real_` numérico
  no R) e rownames viram `_row` → loader corrige (`_coerce_numeric` + `_row`→índice).
  185 testes, ruff limpo, 13 notebooks verdes.

> Fim do Ciclo 1: todo o `tspredit` portado **exceto** EMD/wavelet e o `loadfulldata`
> remoto (as únicas peças com dep pesada).

#### Ciclo 2 — dependências pesadas (extras) + um ++ nativo — **DESENHADO (2026-11-23, Eduardo)**

Uma **família de dep por release**; cada classe faz import preguiçoso + erro acionável
nomeando o extra. Ordem/escopo decididos:

- **✅ 0.7.0 — Filtros de decomposição (EMD) — NATIVO, back-implementation ++**
  *(0 dep nova)* — **IMPLEMENTADO** (branch `milestone-0.7.0-emd`): **PIVÔ decidido com o
  Eduardo (2026-11-23)** — em vez de embrulhar PyEMD (extra `[emd]`), **back-implementamos
  EMD/EEMD nativo** em `emd.py` (numpy+scipy, ambos já no núcleo → **0 dep**, cai o extra
  `[emd]`). `TsFilEmd` (remove 1º IMF; engine `emd` default ou `eemd` opcional) + `TsFilRemd`
  (roughness por IMF → `FitCurvatureMin` + `max_imfs`) em `filters.py`. **Evidência (§1b):**
  nativo é ~**60× mais rápido que o `hht`** do R (EEMD) / **~1800×** (EMD puro), com
  **reconstrução perfeita** (Σ IMF = sinal, 4e-16) e **denoising ≥ PyEMD** (R² 0.971 vs
  0.963). **++ honesto = redução de dep + velocidade + canonicalização** (Python dono; R
  backporta e **dropa `hht`**), **NÃO** um algoritmo novo: o candidato da literatura
  **CEEMDAN foi avaliado e REJEITADO pelo gate** (não melhora o denoising dos filtros, só
  custa 2.7×) — protótipo guardado para se um uso futuro (interpretar IMFs) justificar.
  **Paridade committada:** `develop/parity/filters_emd.R` (referência `hht`) +
  `tests/test_quality_emd.py` — invariante de **qualidade** (melhora de R² + remove o
  outlier **tão bem ou melhor** que o `hht`), não valor-a-valor (EMD é estocástico/backend).
  PyEMD/`hht` ficam **só na validação (dev)**, nunca no runtime. 194 testes, ruff limpo,
  notebook `filter/02-emd-filters`.
- **✅ 0.8.0 — Wavelet — NATIVO, back-implementation** *(0 dep nova)* — **IMPLEMENTADO**
  (branch `milestone-0.8.0-wavelet`): **PIVÔ (Eduardo, 2026-11-23)** — como no EMD, em vez
  de embrulhar PyWavelets (extra `[wavelet]`), **MODWT nativo em `wavelet.py` (numpy puro —
  nem scipy → 0 dep, cai o extra)**. `TsFilWavelet` (zera D1 e reconstrói; `fit` seleciona o
  melhor filtro por R²) sobre ele; famílias haar/d4/la8/bl14/c6 (coef. do R `wavelets`,
  hardcoded, precisão plena). **Diferença vs EMD:** MODWT é transformada linear
  **determinística** → paridade **valor-a-valor ~5e-6** com o R `wavelets` (muito mais
  fiel que o `pywt.swt`, que tem convenção diferente; e o PyWavelets nem está instalado).
  Insight: `ts_fil_wavelet` = `x − D1`, independente do nº de níveis. **++ honesto = redução
  de dep + exatidão + canonicalização** (Python dono; R backporta e **dropa `wavelets`**);
  **thresholding (Donoho-Johnstone) avaliado e REJEITADO pelo gate** (R² 0.25 vs 0.97 do
  zero-D1 — MODWT redundante exige formulação translation-invariant, pesquisa sem ganho
  garantido). c6 reconstrói só a ~1e-6 (coef. coiflet publicados são aproximados, igual no R).
  **Paridade committada:** `develop/parity/filters_wavelet.R` + `tests/test_quality_wavelet.py`
  (valor-a-valor ≤1e-5 vs R, R-free no teste). 203 testes, ruff limpo, notebook
  `filter/03-wavelet-filter`.
- **✅ 0.9.0 — Datasets completos (`loadfulldata`) — 0 DEP, no NÚCLEO** — **IMPLEMENTADO**
  (branch `milestone-0.9.0-loadfulldata`): **PIVÔ (Eduardo, 2026-11-23)** — em vez do extra
  `[bench]=rdata`, **CSV.gz/JSON.gz pandas-nativo, 0 dep**. Benchmark real decidiu: CSV puro
  ~9× o `.RData` (grande demais), mas **CSV.gz ≈ `.RData`** (NN5: RData 157KB / CSV 1422KB /
  **CSV.gz 352KB** / parquet 693KB) — e **parquet perdeu em tamanho E exige `pyarrow`**.
  `gzip` é stdlib e `pd.read_csv(url, compression="gzip")` é pandas-núcleo → **sem
  `rdata`/`xarray`/`pyarrow`, cai o extra `[bench]`**. `datasets.loadfulldata(nome)` =
  `urllib`(stdlib) baixa a `full_url` do manifest → pandas/gzip → MESMA estrutura do loader
  da miniatura. **ETL do tspredbench estendido** (`02-build-minidata.R`): publica os dados
  completos como `.csv.gz`/`.json.gz` em `data/python_full/` + `full_url` no manifest
  (branch tspredbench `full-data-python`). Miniaturas seguem CSV/JSON puro. **Paridade:
  miniatura = head(30) do full (exato por construção).** Teste via `file://` (sem rede, R-free);
  o fetch remoto fica ativo quando o tspredbench full-gz for para a main. 207 testes, ruff
  limpo. **Terceiro back-implementation/redução-de-dep do Ciclo 2 — o roadmap inteiro ficou
  sem deps novas (hht, wavelets, rdata todos eliminados).**
- **✅ 0.10.0 — ARIMA++ (auto-seleção nativa)** *(0 dep nova)* — **IMPLEMENTADO** (branch
  `milestone-0.10.0-autoarima`): **primeiro ++ ALGORÍTMICO de verdade** (EMD/wavelet foram
  redução-de-dep). `TsArima(auto="cv")` ataca 2 fraquezas do `auto.arima`: (1) **candidatos
  dirigidos por ACF/PACF** — os lags significativos entram na busca (lag sazonal 12 testado
  quando os dados pedem) em vez do cap fixo 5; (2) **seleção por CV rolling-origin casada ao
  horizonte** (sMAPE out-of-sample), não o AICc in-sample. Tudo nativo (KPSS/acf/pacf/ARIMA já
  statsmodels-núcleo); `auto="aicc"` segue como default p/ compatibilidade. **Gate §1b PASSOU:
  no NN5 (dados reais de competição) o cv++ é ~14% melhor que o aicc** (sMAPE 0.1352 vs 0.1565,
  ganha 4/4, recupera o lag semanal 7). **Achado-chave:** a CV precisa **casar o horizonte da
  previsão** — CV de 1-passo escolhia ordens que generalizavam mal (0.1655); multi-step casado
  deu 0.1352. Em série AR(12) sintética também ganha (0.19 vs 0.24, pega lag ~12). Benchmark
  committado em `develop/parity/autoarima_benchmark.py`. Testes rápidos (componentes + e2e
  pequeno; o "ganha" fica no benchmark manual, não na suíte). Notebook `prediction/04-arima-plus`.

> **Fim do Ciclo 2 — COMPLETO (0.7→0.10).** Todo o `tspredit` portado + o ARIMA++.
> **Resultado: o Ciclo 2 NÃO adicionou nenhuma dep** — EMD (numpy+scipy), wavelet (numpy
> puro), `loadfulldata` (CSV.gz pandas+stdlib) back-implementados 0-dep (extras
> `[emd]`/`[wavelet]`/`[bench]` eliminados); e o ARIMA++ (0 dep) foi o **único ++
> algorítmico**, validado contra o auto.arima. Depois vêm as passadas de maturidade
> (§5.4/Maturidade) e a 1.0.0.

#### Auditoria de imports + candidatos ++ / redução de dep (2026-11-23)

> Pedida pelo nextprompt (fazer antes de escolher o próximo marco).

**Imports de terceiros no núcleo:** `numpy`, `pandas`, `scipy`, `scikit-learn`
(`KNeighborsRegressor`/`MLPRegressor`/`RandomForestRegressor`/`SVR`/`StandardScaler`/
`TransformedTargetRegressor`/`make_pipeline`), `statsmodels` (`ARIMA`, `lowess`,
`ExponentialSmoothing`, `STL`, `acf`/`pacf`/`kpss`), `matplotlib`, `pydaltoolbox`.
**Todas canônicas** — nenhuma dep atípica/nicho no núcleo → **nada a back-implementar por
atrito**; todas seguem embrulhadas. As únicas deps de nicho (PyEMD/PyWavelets/`rdata`) já
estão isoladas como **extras** (padrão correto).

**Candidatos ++ (caso a caso, não regra cega):**
1. **Auto-seleção ARIMA nativa** → **FEITO na 0.10.0** (ARIMA++ `auto="cv"`: candidatos
   ACF/PACF + CV casada ao horizonte). **Único ++ algorítmico** do Ciclo 2 e **passou o gate**
   (~14% melhor que o aicc/auto.arima no NN5). (ETS no pool fica p/ uma eventual extensão futura.)
2. **EMD nativo** → **FEITO na 0.7.0** (numpy+scipy, 0 dep; ~60× mais rápido que `hht`,
   denoising ≥ PyEMD). Elimina o extra `[emd]`; R dropa `hht`. CEEMDAN avaliado e rejeitado
   pelo gate (sem ganho para o uso dos filtros). **Exemplo concreto do braço de redução-de-dep
   do §1b dando certo.**
3. **MODWT/wavelet nativo** → **FEITO na 0.8.0** (numpy puro, **0 dep, nem scipy**;
   determinístico → paridade valor-a-valor ~5e-6 com R). Elimina o extra `[wavelet]`/PyWavelets;
   R dropa `wavelets`. Thresholding avaliado e rejeitado pelo gate. **Segundo back-implementation
   bem-sucedido (ainda mais forte: 0 dep total + quase-exato).**
4. Demais (sklearn/scipy/statsmodels) **permanecem embrulhados** — maduros e com diferença
   pequena (ex.: MLP não vira back-implementation, como no pydaltoolbox).

#### Maturidade — espelha o pydaltoolbox 0.20→0.24 (decidido 2026-11-23)

Agora que o Ciclo 2 fechou (todo o `tspredit` portado), rodamos as MESMAS passadas de
maturidade que elevaram o pydaltoolbox de "portado" para "publicável como referência".
Mapeamento passo-a-passo (pydaltoolbox → pytspredit):

- **M1 — Organização de docs (espelha pydaltoolbox 0.20, só-R).** O `tspredit` R já usa
  `notebooks/`→`examples/` (não é `Rmd/`); passada leve: confirmar o `conversor.R`, regenerar
  `examples/` do pacote atual (sem `hht`/`wavelets`/`forecast`), remover ativos órfãos. Só-R,
  sem release PyPI.
- **✅ M2 — Exemplos 1:1 R↔Python (espelha 0.21/0.22) — FEITO** (branch `milestone-m2-examples`):
  ~103 notebooks um-por-classe em 11 famílias, todos verdes (Pacote 1: tsdata 3, normalization 9,
  lagmapping 13, augment 8, filter 16, prediction 11, tuning 2, multivariate 6; Pacote 2: datasets
  21, custom 4, tutorial 10). Torch fora de escopo; variantes stock-close adiadas (sem classe nova).
- **M2 (descrição original) — Exemplos 1:1 R↔Python (espelha 0.21/0.22).** Reorganizar os notebooks do pytspredit
  para **um por classe portada**, espelhando a árvore e a didática do R (About / Didactic goal
  / setup / load / split / fit / eval / What to observe / References). Famílias: `tsdata`,
  `normalization`, `lagmapping`, `prediction`, `filter`, `augment`, `multivariate`, `tuning`,
  `datasets`, `tutorial`, `custom`. **Fora de escopo:** torch (conv1d/lstm/torch-ts-mlp).
  Meta: **paridade de contagem família-a-família** com o R (como o 107=107 do pydaltoolbox).
  **Exemplos do ARIMA++/auto curtos** (sem busca lenta nos que rodam). Dev-only (sem PyPI),
  provavelmente 1–2 passadas.
- **M3 — Testes pareados R↔Python retroativos (espelha 0.23 + o ciclo de paridade).**
  Consolidar `tests/test_quality_*.py` + scripts em `develop/parity/` cobrindo 0.1→0.5
  (normalização = **matriz igual** ≤1e-9; lagmap = conjuntos; regressores = R²/métricas;
  deterministic/darima/warma; filtros; multivariado). Congelar o input em CSV; comparar o
  invariante da família; **achar e corrigir bugs nos dois lados** (como os 6 do pydaltoolbox).
  Fix de código → pode gerar release pequeno; resto dev-only.
- **M4 — Metadados (espelha "Metadados do pacote").** `pyproject.toml`: contribuidores
  (espelhar o `Authors@R` do `tspredit` R: Eduardo, Arthur Garcia, Cristiane Gea, Diogo
  Santos, …), e **Development Status `3 - Alpha` → `4 - Beta`**.
- **✅ M5 — `parity.md` (espelha "Metodologia de paridade") — FEITO** (branch
  `milestone-m5-parity-matrix`, dev-only): §5 estudos de caso (ts_norm_none, ts_arimax xreg,
  ARIMA++/horizonte-CV, EMD degenerado, read-only numpy) + §6 matriz de paridade R↔Py
  completa e autoritativa (invariante/veredito/evidência/guard por classe). `mapeamento.md`
  aponta a §6 como matriz autoritativa.
- **✅ M6 — Auditoria de crédito (espelha 0.24.0), Python + R — FEITO** (branch
  `milestone-m6-credit-audit` no pytspredit + `m6-credit-audit` no tspredit). Varredura de
  TODAS as docstrings (Python) + roxygen (R): **frases proibidas = ZERO** (removidas de
  arima/deterministic/emd/wavelet/filters/regressors/multivariate no Python e de
  ts_arima/ts_fil_emd/ts_fil_remd/ts_fil_wavelet/tslagutils no R — a remoção de dep vai na
  mensagem de commit, não na doc); Teoria + `References:` + `Porting:` nas classes de
  algoritmo (adicionado `Theory:` em EMD/wavelet/ARIMA/ELM/deterministic); **atribuição
  canônica das back-impls/++** com evidência (EMD ~1800× mais rápido; wavelet valor-a-valor
  ~5e-6; ARIMA++ ~14% melhor no NN5; `_RLinear`=`lm` do R; ELM; `TsVar`); corrigidas refs
  `Mirrors X.R` inexistentes (`ts_periodic.R`/`ts_persist.R`→`ts_deterministic.R`;
  `grf_ts_pred.R` confirmado como arquivo do daltoolbox). Python: 261 testes verdes, ruff
  limpo. R: roxygenise 8.1.1 regenerou 6 `.Rd`, `R CMD INSTALL` limpo, versão mantida em
  2.0.747 (polimento de doc da release ainda não submetida ao CRAN). Dev-only.

#### Fechamento — BETA primeiro, depois 1.0.0 (decidido 2026-11-23; **beta renomeado 2026-11-24**)
- **⚠️ MUDANÇA (Eduardo, 2026-11-24):** o beta **NÃO** será `1.0.0b1`. Vamos **batizar de
  beta a versão subsequente à última de qualidade**, e ela **continua na linha `0.x.0`**
  (ex.: `0.12.0`), não salta para `1.0.0b1` (PEP 440). Ou seja: as passadas de maturidade
  (M1–M6 + a tarefa extra que o Eduardo vai passar antes do beta) são "de qualidade"; a
  PRÓXIMA `0.x.0` depois delas é "o beta".
- **⏸️ ANTES DO BETA:** o Eduardo vai passar **outra tarefa** — fazer essa tarefa primeiro,
  depois a release beta.
- **⬜ `0.x.0` — RELEASE BETA.** Fechadas M1–M6 + a tarefa extra, publicar a próxima `0.x.0`
  no PyPI como o beta — `mapeamento.md`/`historico.md` finalizados, matriz de paridade
  completa, API revisada. **Rodar ~1 semana em beta**, coletar feedback.
- **⬜ `1.0.0` — estável.** Após a semana de beta sem problemas, promover a `1.0.0`
  (congela a API, sai de beta). **A partir daqui o pytspredit também usa numeração Boeing**
  (1.0.707, 1.0.717, …), como o `tspredit` R.

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
- [x] **Sub-decisão (Ciclo 2, datasets):** formato das miniaturas e leitor RData.
      ✅ (2026-11-23) Miniaturas = CSV/JSON+manifest (resolvido na 0.6.0); leitor do
      `loadfulldata` = **`rdata`** (puro-Python), só no extra `[bench]` (0.9.0).
- [x] **Ciclo 2 DESENHADO com o Eduardo** (2026-11-23): 0.7 EMD `[emd]` → 0.8 wavelet
      `[wavelet]` → 0.9 `loadfulldata` `[bench]` → **0.10 ++ auto-seleção ARIMA/ETS nativa**
      → maturidade → 1.0.0. Uma família de dep por release; EMD/wavelet embrulham
      PyEMD/PyWavelets com tolerância documentada. Ver §5.4/Ciclo 2.

### Resolvidas
- [x] **Maturidade espelha o pydaltoolbox 0.20→0.24; fecho com BETA antes da 1.0.0**
      (2026-11-23, Eduardo): 6 passos M1–M6 (docs → exemplos 1:1 → testes pareados
      retroativos → metadados/Beta classifier → parity.md → auditoria de crédito Py+R);
      o **último passo é um release `1.0.0b1`** rodando **~1 semana** em beta, depois a
      `1.0.0` estável. Pós-1.0.0 o pytspredit também adota numeração Boeing. Ver §5.4.
- [x] **Práticas de maturidade do pydaltoolbox adotadas** (2026-11-23, Eduardo): criar
      `historico.md`/`mapeamento.md`/`parity.md` **agora** (scaffold); **ritual de paridade
      committada** (`develop/parity/` + `tests/test_quality_*`) a partir do **0.6.0**;
      **exemplos 1:1** (um por classe portada) e **auditoria de crédito** (Python **+** R)
      como **passadas finais** perto da 1.0.0; famílias **torch fora de escopo**. Ver §1c.
- [x] Backend do ELM (M3): implementação própria (random hidden uniform_positive + pinv,
      sem bias). Com `actfun="purelin"` e `nhid >= nº de lags` equivale a OLS → bate com
      o R independente do RNG. Com `nhid` menor, projeção aleatória (diverge por RNG).
- [x] `TsArima` sobre statsmodels: ordem manual `(p,d,q)` fiel; `auto` via busca AICc
      (aproxima `forecast::auto.arima`, pode não reproduzir a ordem exata). Backend MLE
      difere do `forecast` → previsões próximas, não bit-idênticas.
- Nota: `TsArimax` é multivariado (xreg) → movido para o Bloco B junto da família MV.
- [x] **Fecho M4:** build + `twine check` OK; sdist exclui `develop/notebooks/examples/tests`
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
