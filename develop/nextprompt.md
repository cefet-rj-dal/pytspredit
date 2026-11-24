Retomada do pytspredit (c:\Git\dal\pytspredit). Leia primeiro a memória
[[pytspredit-port]], develop/planejamento.md (§1c + §5.4/Maturidade), develop/historico.md,
develop/mapeamento.md e develop/parity.md. Estamos na **reta de maturidade rumo à 1.0.0**,
espelhando o pydaltoolbox 0.20→0.24, e o **último passo é um release BETA `1.0.0b1`**
(rodar ~1 semana) — NÃO ir direto à 1.0.0.

ESTADO ATUAL (2026-11-24)
- **pytspredit 0.11.0 PUBLICADO no PyPI** (https://pypi.org/project/pytspredit/0.11.0/) e na
  main (HEAD inclui `fit-arima-export` ff-mergeada). **Ciclo 2 completo** — todo o `tspredit`
  portado, 0 dep nova.
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
- **M3 ✓ FEITO** na branch **`milestone-m3-paired-tests`** (pytspredit, pushed, dev-only,
  aguardando ff-merge): 6 pares `develop/parity/<fam>.R`→`<fam>_ref.json` + guards R-free
  `tests/test_quality_<fam>.py` para normalização/lagmap/regressores/model-zoo/filtros/
  multivariado (0.1→0.5); datasets/emd/wavelet já tinham. **Achou+corrigiu 1 bug REAL do R**:
  `ts_arimax` predict quebrava (`object 'xreg' not found`) — fix em `arima_fit`
  (`fit$call$xreg <- xreg`), na branch **`arimax-xreg-fix`** do tspredit (pushed, DESCRIPTION
  **2.0.737→2.0.747**). Python já estava certo.
- Branches de feature absorvidas e DELETADAS (local+remoto): `fit-arima-export`,
  `emd-native-backport`, `milestone-m2-examples`.

PENDÊNCIAS DE MERGE
- ff-merge de `milestone-m2-examples`? NÃO — já estava na main. Pendentes agora:
  **`milestone-m3-paired-tests`** (pytspredit, dev-only ff) e **`arimax-xreg-fix`**
  (tspredit, ff→main). Push direto na main do tspredit é bloqueado pelo classificador → use branch.

PENDÊNCIA EXTERNA (Eduardo faz)
- submissão ao CRAN do `tspredit` **v2.0.747** (após ff-merge do `arimax-xreg-fix`; o alvo
  mudou de 2.0.737 por causa do fix do ARIMAX achado no M3).

PRÓXIMOS PASSOS DE MATURIDADE (ver §5.4/Maturidade) — faça na ordem, cada um numa branch:
- **M4 — Metadados:** `pyproject.toml` com contribuidores (espelhar o `Authors@R` do
  `tspredit` R) e **Development Status `3 - Alpha` → `4 - Beta`**.
- **M5 — `parity.md`:** estudos de caso do M3 + matriz de paridade R↔Py completa.
- **M6 — Auditoria de crédito (espelha 0.24.0), Python + R:** zero frases proibidas; Teoria +
  `References:` + `Porting:` em toda classe de algoritmo; atribuição canônica das
  back-implementations/++ (EMD, wavelet, ARIMA++, `_RLinear`, ELM, `TsVar`); corrigir
  `Mirrors X.R` a arquivos inexistentes. Espelhar na roxygen do `tspredit`.
- **FECHO — `1.0.0b1` (BETA):** publicar a versão beta (PEP 440), rodar ~1 semana, coletar
  feedback; depois promover a **1.0.0** estável. **Pós-1.0.0 o pytspredit também adota
  numeração Boeing** (1.0.707, 1.0.717, …), como o `tspredit` R.

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

COMECE confirmando comigo: fazer os ff-merges pendentes (M3 no pytspredit, `arimax-xreg-fix`
no tspredit) e seguir para o **M4** (metadados / classificador Beta) — ou priorizar outro
passo da reta de maturidade (M5 parity.md / M6 auditoria de crédito) rumo ao `1.0.0b1` beta.
