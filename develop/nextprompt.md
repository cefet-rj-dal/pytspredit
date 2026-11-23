Vamos implementar a 0.7.0 do pytspredit (c:\Git\dal\pytspredit). Leia primeiro a
memória [[pytspredit-port]], develop/planejamento.md (§5.4, Ciclo 2, bloco 0.7.0) e
develop/historico.md (0.6.0 fechou o Ciclo 1) — 0.1.0→0.6.0 já estão no PyPI e tudo
ff-merged na main (só a branch main existe).

ESCOPO 0.7.0 — "Filtros de decomposição (EMD)" — PRIMEIRA release do Ciclo 2:
- `TsFilEmd` (porta `ts_fil_emd.R`): decompõe em IMFs e remove o 1º IMF como ruído
  (`result = data - imf1`). Params `noise=0.1`, `trials=5`.
- `TsFilRemd` (porta `ts_fil_remd.R`): roughness por IMF (1ª diferença normalizada) →
  corte via `FitCurvatureMin` do pydaltoolbox sobre a roughness cumulativa, com cap
  `max_imfs`; remove os IMFs de alta frequência selecionados.
- Ambos em `filters.py`, sobre a série 1-D (DalTransform), no mesmo padrão dos 15
  filtros existentes (References:/Porting:).
- **Primeiro EXTRA opcional do pytspredit:** `[emd]` = **PyEMD** (`pip install
  pytspredit[emd]`; pacote PyPI `EMD-signal`, import `PyEMD`). Import PREGUIÇOSO dentro
  do método, com erro acionável nomeando o extra se faltar (padrão do pydaltoolbox).
  PyEMD NÃO entra no núcleo.

DECISÕES JÁ TOMADAS (2026-11-23, não reabrir sem motivo):
- Embrulhar PyEMD (não back-implementar) + documentar tolerância. Mapear
  `hht::CEEMD`(noise, trials) → PyEMD **EEMD** (`noise_width`, `trials`); `random_state`
  para reprodutibilidade.
- Paridade R vs Py = **tolerância de backend/estocástica** (EMD é ensemble; `hht` ≠
  PyEMD). O invariante NÃO é valor-a-valor: usar forma/estrutura do reconstruído e R²
  (ex.: o reconstruído explica bem a série; remove ruído sem distorcer a tendência).
  Documentar a tolerância como em qes/seas_adj/spline (0.4.0).

RITUAL OBRIGATÓRIO (igual às releases anteriores):
- Branch por milestone (milestone-0.7.0-emd); exemplos sempre commitados; me passar os
  links do GitHub ao final; ff-merge na main só quando eu pedir; o twine upload eu
  autorizo/rodo.
- **Paridade committada (ritual desde 0.6.0):** script em `develop/parity/` (ex.:
  `filters_emd.R` gerando referência a partir do `tspredit` R com `hht`) + guarda em
  `tests/test_quality_*.py` (tolerância frouxa para o estocástico — "não degenera",
  "R² acima de baseline", "remove o IMF certo"). R 4.5.0 em /c/R/R-4.5.0, pacote R em
  c:\Git\dal\tspredit (tem `ts_fil_emd`/`ts_fil_remd` via `hht`).
- Convenção de doc (travada): toda classe nova com References: (do @references do R) +
  Porting: (arquivo R + nível exact/approximate/adapted + adaptações).
- Verde sempre: ruff check . + pytest + python notebooks/conversor.py; um notebook
  didático (notebooks/filter/… ou notebooks/decomposition/…) + testes unitários; bump de
  versão em pyproject.toml e src/pytspredit/__init__.py; declarar o extra `[emd]` em
  [project.optional-dependencies]; atualizar README, planejamento.md, historico.md,
  mapeamento.md (marcar TsFilEmd/TsFilRemd ✅ 0.7.0).

COMECE decidindo comigo as sub-decisões em aberto antes de codar:
(1) algoritmo PyEMD exato (EEMD vs CEEMDAN) e os defaults que melhor refletem
    `hht::CEEMD`; (2) como expor `random_state` nos dois filtros; (3) o invariante
    committado de paridade para o estocástico (qual métrica/tolerância). Proponha e
    aguarde meu ok.

DEPOIS do 0.7.0 (Ciclo 2, já desenhado — ver planejamento §5.4): 0.8.0 wavelet
`[wavelet]` (PyWavelets) → 0.9.0 `loadfulldata` `[bench]` (`rdata`) → 0.10.0 ++
auto-seleção ARIMA/ETS nativa → passadas de maturidade (exemplos 1:1 + auditoria de
crédito Py+R) → 1.0.0.
