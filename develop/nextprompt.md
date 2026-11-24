Retomada do pytspredit (c:\Git\dal\pytspredit). Leia primeiro a memória
[[pytspredit-port]], develop/planejamento.md (§1c + §5.4/Maturidade), develop/historico.md,
develop/mapeamento.md e develop/parity.md. Estamos na **reta de maturidade rumo à 1.0.0**,
espelhando o pydaltoolbox 0.20→0.24, e o **último passo é um release BETA `1.0.0b1`**
(rodar ~1 semana) — NÃO ir direto à 1.0.0.

ESTADO ATUAL (2026-11-24)
- **pytspredit 0.11.0 na main** (HEAD `1d0101d`, ff-merge de `fit-arima-export`) — **ainda
  NÃO publicado no PyPI** (0.10.0 é o último no PyPI; `twine upload` do 0.11.0 aguarda o
  "pode fazer"). **Ciclo 2 completo** — todo o `tspredit` portado, 0 dep nova.
- **`fit_arima` exportado nos DOIS lados (0.11.0)**: a auto.arima nativa (sem `forecast`)
  virou função pública `fit_arima(series, auto="aicc"|"cv", ...)` (Python `pytspredit.fit_arima`
  + R `fit_arima()` com `@export`/NAMESPACE/Rd). Novo gatilho ADITIVO: `TsArima(0,0,0)` /
  `ts_arima(0,0,0)` dispara auto, junto com `None`/`NULL` (que continua default). Só o (0,0,0)
  cheio é sentinela; (0,1,1) etc. seguem manuais. Nome `fit_arima` (não `fit.arima`: `.` é
  inválido em Python e colidiria com S3 `fit` em R).
- **Backport R MESCLADO NA MAIN** do `tspredit` (HEAD `130e7f3`, v2.0.737): EMD + wavelet +
  ARIMA++ nativos + `fit_arima`; **`hht`, `wavelets` e `forecast` droppados**. Falta só a
  **submissão ao CRAN** (passo manual do Eduardo).
- **M1 ✓ e M2 ✓ (exemplos 1:1)** na branch **`milestone-m2-examples`** (pushed, dev-only,
  aguardando ff-merge): ~103 notebooks um-por-classe em 11 famílias, todos verdes
  (torch fora de escopo; variantes stock-close adiadas).
- Branches `fit-arima-export` (pytspredit) e `emd-native-backport` (tspredit) já absorvidas
  na main — podem ser deletadas.

PENDÊNCIAS (Eduardo decide; eu executo quando ele pedir)
1. **`twine upload` do pytspredit 0.11.0** (build + `twine check` posso rodar; o upload
   aguarda o "pode fazer").
2. ff-merge da `milestone-m2-examples` na main do pytspredit (dev-only, sem release PyPI —
   como o 0.21/0.22 do pydaltoolbox). Depois deletar a branch.
3. submissão ao CRAN do `tspredit` v2.0.737 (já na main do R).
4. deletar as branches já mescladas (`fit-arima-export`, `emd-native-backport`), se quiser.

PRÓXIMOS PASSOS DE MATURIDADE (ver §5.4/Maturidade) — faça na ordem, cada um numa branch:
- **M3 — Testes pareados R↔Python retroativos** (espelha o 0.23 + o ciclo de paridade do
  pydaltoolbox): consolidar `tests/test_quality_*.py` + scripts em `develop/parity/` cobrindo
  0.1→0.5 (normalização = MATRIZ igual ≤1e-9; lagmap = conjuntos; regressores = R²/métricas;
  deterministic/darima/warma; filtros; multivariado). Congelar input em CSV, comparar o
  invariante da família, **achar e corrigir bugs nos dois lados**. Fix de código → release
  pequeno; resto dev-only.
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

COMECE confirmando comigo: publicar o 0.11.0 no PyPI (twine) e/ou fazer o ff-merge do M2, e
seguir para o M3 — ou priorizar outro passo.
