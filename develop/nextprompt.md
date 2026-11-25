Retomada do pytspredit (c:\Git\dal\pytspredit). Leia primeiro a memória
[[pytspredit-port]], develop/planejamento.md (§1c + §5.4/Maturidade), develop/historico.md,
develop/mapeamento.md e develop/parity.md. Estamos na **reta de maturidade rumo à 1.0.0**,
espelhando o pydaltoolbox 0.20→0.24. **MUDANÇA (Eduardo, 2026-11-24): o beta NÃO é `1.0.0b1`**
— vamos **batizar de beta a próxima `0.x.0`** (a versão subsequente à última de qualidade),
continuando na linha 0.x; roda ~1 semana → promove a 1.0.0. **M1–M6 ✓.** **TAREFA EXTRA
pré-beta (migração dos plots de TS) ✓ FEITA e PUBLICADA (pytspredit 0.12.0 no PyPI).**
**PRÓXIMO = a release BETA `0.x.0`** (quando o Eduardo der o OK).

ESTADO ATUAL (2026-11-24)
- **pytspredit 0.11.1 PUBLICADO no PyPI** (https://pypi.org/project/pytspredit/0.11.1/) e na
  main (HEAD `3dd8c71`). 0.11.1 = release interino só de metadados do M4 (autores do port +
  Development Status **4 - Beta**), sem mudança funcional desde 0.11.0. **Ciclo 2 completo** —
  todo o `tspredit` portado, 0 dep nova.
- **`fit_arima` exportado nos DOIS lados (0.11.0)**: a auto.arima nativa (sem `forecast`)
  virou função pública `fit_arima(series, auto="aicc"|"cv", ...)` (Python `pytspredit.fit_arima`
  + R `fit_arima()` com `@export`/NAMESPACE/Rd). Gatilho ADITIVO: `TsArima(0,0,0)` /
  `ts_arima(0,0,0)` dispara auto, junto com `None`/`NULL` (que continua default). Só o (0,0,0)
  cheio é sentinela; (0,1,1) etc. seguem manuais. Nome `fit_arima` (não `fit.arima`: `.` é
  inválido em Python e colidiria com S3 `fit` em R).
- **Backport R MESCLADO NA MAIN** do `tspredit` (HEAD `130e7f3`, v2.0.737): EMD + wavelet +
  ARIMA++ nativos + `fit_arima`; **`hht`, `wavelets` e `forecast` droppados**. Falta só a
  **submissão ao CRAN** (passo manual do Eduardo — ele mesmo faz).
- **M1 ✓ e M2 ✓ (exemplos 1:1) JÁ NA MAIN**: 104 notebooks/exemplos um-por-classe em 11
  famílias (tsdata/normalization/lagmapping/augment/filter/prediction/tuning/multivariate/
  datasets/custom/tutorial), todos verdes (torch fora de escopo; stock-close adiadas). A
  branch `milestone-m2-examples` já foi ff-mergeada e deletada em sessão anterior.
- **M3 ✓ FEITO E MESCLADO NA MAIN** (pytspredit HEAD `28e8516`, dev-only sem PyPI): 6 pares
  `develop/parity/<fam>.R`→`<fam>_ref.json` + guards R-free `tests/test_quality_<fam>.py`
  para normalização/lagmap/regressores/model-zoo/filtros/multivariado (0.1→0.5);
  datasets/emd/wavelet já tinham. **Achou+corrigiu 1 bug REAL do R**: `ts_arimax` predict
  quebrava (`object 'xreg' not found`) — fix em `arima_fit` (`fit$call$xreg <- xreg`),
  **MESCLADO na main do tspredit** (HEAD `e5f4583`, DESCRIPTION **2.0.747**). Python já certo.
- Branches de feature absorvidas: `fit-arima-export`, `emd-native-backport`,
  `milestone-m2-examples` (deletadas); `milestone-m3-paired-tests` (pytspredit) e
  `arimax-xreg-fix` (tspredit) já mescladas — PODEM SER DELETADAS. `main` é a única de código.

PENDÊNCIA EXTERNA (Eduardo faz)
- submissão ao CRAN do `tspredit` **v2.0.747** (já na main do R; o alvo mudou de 2.0.737
  por causa do fix do ARIMAX achado no M3).

PRÓXIMOS PASSOS DE MATURIDADE (ver §5.4/Maturidade) — faça na ordem, cada um numa branch;
**o próximo é o M5**:
- **M4 ✓ FEITO E MESCLADO NA MAIN** (branch deletada): `pyproject.toml` com
  `maintainers`/`authors` dos contribuidores do PORT Python (subconjunto distinto do
  `Authors@R` do R — Eduardo corrigiu: NÃO espelhar o R cegamente; Rebecca com email INRIA) +
  **Development Status `3 - Alpha` → `4 - Beta`**. Publicado no 0.11.1 (antecipado a pedido do
  Eduardo; o beta `1.0.0b1` segue depois do M5+M6). Lista autoritativa: memória [[pytspredit-authors]].
- **M5 ✓ FEITO** (branch `milestone-m5-parity-matrix`, pushed, dev-only, AGUARDANDO
  revisão/merge do Eduardo): `parity.md` ganhou **§5 estudos de caso** (ts_norm_none,
  ts_arimax xreg, ARIMA++/horizonte-CV, EMD degenerado, read-only numpy) + **§6 matriz de
  paridade R↔Py completa e autoritativa** (invariante/veredito/evidência/guard por classe);
  `mapeamento.md` aponta a §6. Doc-only, ruff limpo, sem mudança de código.
- **M6 ✓ FEITO** (pytspredit branch `milestone-m6-credit-audit` + tspredit branch
  `m6-credit-audit`, pushadas, dev-only): auditoria de crédito Py+R — frases proibidas=0
  (removidas das docstrings/roxygen; dep-removal vai no commit, não na doc), Theory +
  `References:` + `Porting:` nas classes de algoritmo, atribuição canônica das back-impls/++
  (EMD/wavelet/ARIMA++/`_RLinear`/ELM/`TsVar`) com evidência, refs `Mirrors X.R` corrigidas.
  Python 261 testes verdes + ruff; R roxygenise+`R CMD INSTALL` limpos (2.0.747 mantida).
- **TAREFA EXTRA (migração dos plots de TS) ✓ FEITA + PUBLICADA (2026-11-25):** os plots de
  série temporal (`plot_ts`/`plot_ts_pred`/`plot_ts_pred_mv`/`plot_series`) saíram do
  daltoolbox/pydaltoolbox e viraram do pytspredit/tspredit (ver [[pytspredit-port]] +
  develop/plano-plots.md). **pytspredit 0.12.0** e **pydaltoolbox 0.25.0** no PyPI;
  **daltoolbox 1.4.717** e **tspredit 2.0.757** nas mains (CRAN = Eduardo). Exemplos Python
  70/104 com plot. daltoolboxdp/reproduce ficaram de fora (migrar depois).
- **FECHO — BETA = próxima `0.x.0`** (não `1.0.0b1`): publicar a próxima `0.x.0` como o beta,
  rodar ~1 semana, coletar feedback; depois promover a **1.0.0** estável. **Pós-1.0.0 o
  pytspredit também adota numeração Boeing** (1.0.707, 1.0.717, …), como o `tspredit` R.

RITUAL (igual às releases anteriores)
- Branch por milestone; exemplos sempre committados; me passar os links do GitHub ao final;
  ff-merge na main só quando eu pedir; **o `twine upload` eu autorizo** (nota: o classificador
  do harness bloqueia o `twine upload` até eu dar o "pode fazer"; o build + `twine check`
  podem rodar normalmente).
- Verde sempre: `ruff check .` + `pytest` + `python notebooks/conversor.py` (o conversor
  aceita UM argumento: uma família ou um arquivo).
- Paridade committada a cada passo (`develop/parity/` + `tests/test_quality_*.py`); R 4.5.0 em
  /c/R/R-4.5.0; pacote R em c:\Git\dal\tspredit.
- Convenção de doc travada: `References:` (do @references do R) + `Porting:` em toda classe.
- **Exemplos do roxygen (R) têm que rodar RÁPIDO** — ordens explícitas, nunca busca
  automática/CV; documentar os modos lentos no `@details`.

GOTCHAS IMPORTANTES DESTA MÁQUINA
- **`library(tspredit)` / `pkgload::load_all` dão SEGFAULT intermitente** (conflito de ABI
  das libs nativas/BLAS, pré-existente, NÃO é o nosso código) — validar o R via
  `R CMD INSTALL` + scripts standalone (eval do arquivo), não via load_all/library.
- `roxygen2` é 8.1.1 (ok, >= 8.1.0 do pacote) → `roxygenise()` regenera NAMESPACE/Rd
  corretamente.
- `TsNormDiff` perde 1 coluna → em exemplos com diff, `input_size < sw-1`.
- ARIMA++ (`TsArima(auto="cv")`) é lento (~15s) — manter fora de exemplos que precisam ser
  rápidos; o benchmark fica em `develop/parity/autoarima_benchmark.py`.

COMECE pela **release BETA = próxima `0.x.0`** (a tarefa extra dos plots já foi feita+publicada
na 0.12.0). Bump pyproject + `__init__`, build, twine check, upload com o "pode fazer" do
Eduardo, roda ~1 semana → promove a 1.0.0.
(M1–M6 ✓, tudo na main; **pytspredit 0.12.0** e **pydaltoolbox 0.25.0** no PyPI; **daltoolbox
1.4.717** e **tspredit 2.0.757** nas mains do R aguardando o teste/submissão ao CRAN do Eduardo.)
