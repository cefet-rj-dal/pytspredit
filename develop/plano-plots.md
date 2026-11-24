# Plano — Responsabilidades dos plots de TS + enriquecimento dos exemplos

Decidido com o Eduardo (2026-11-24). Duas frentes; a B depende da A.

## Decisões (Eduardo)
1. **Mover de fato** (quebra, mais limpo): os plots de série temporal saem do
   `daltoolbox`/`pydaltoolbox` (toolbox geral) e passam a ser responsabilidade do
   **`tspredit`/`pytspredit`** (o pacote de TS). Atualizar os consumidores.
2. **Todos os temporais:** `plot_ts_pred`, `plot_ts_pred_mv`, `plot_ts`, `plot_series`
   (R: `grf_ts_pred.R`, `ts_plot_mv.R`, `grf_ts.R`, `grf_series.R`).
3. **R e Python juntos**, espelhando e verificando paridade a cada passo.

## Estado inicial (auditoria 2026-11-24)
- **R:** `plot_ts`/`plot_ts_pred`/`plot_series` exportados pelo **daltoolbox**
  (`grf_ts.R`/`grf_ts_pred.R`/`grf_series.R`); `tspredit` só tem `plot_ts_pred_mv`
  (`ts_plot_mv.R`) e **já importa `daltoolbox`**.
- **Python:** `plot_ts`/`plot_ts_pred`/`plot_series` no **`pydaltoolbox.viz`**
  (`viz/temporal.py`, extra `[viz]`) **e** `plot_ts_pred`/`plot_ts_pred_mv` no
  **`pytspredit.plot`** (duplicado). `pytspredit.plot` é auto-contido (matplotlib direto,
  retorna `Figure`); `pydaltoolbox.viz.temporal` retorna `Axes` e usa `viz._helpers`.
- **Lacuna dos exemplos:** R faz plot em **83/119** exemplos (~70%); Python em **6/104**
  (~6%). Os exemplos R usam `plot_ts` (série crua) + `plot_ts_pred` (forecast/filtro).

## Raio de impacto (consumidores a migrar)
| Consumidor | R (.R) | Notebooks (.md) | Python (.py) |
|---|---|---|---|
| daltoolbox (exemplos próprios) | 5 | 5 | — |
| daltoolboxdp | 28 | 28 | — |
| pydaltoolbox (notebooks próprios) | — | 12 | 1 (def.) |
| reproduce (código de papers) | 3 | — | 1 |
| tspredit (já `library(tspredit)`) | — | 83 | — |

> **Implicação arquitetural:** `daltoolboxdp`/`harbinger` também plotam TS → ao mover,
> passam a **depender do `tspredit`/`pytspredit`** (sibling→sibling). É a consequência de
> eleger o tspredit como núcleo de TS. **`reproduce/` é código de reprodução de papers —
> confirmar com o Eduardo antes de editar.**

## Fases (staging para não quebrar no meio do caminho)
- **Fase 1 — ADITIVA (segura): os donos ganham as funções.**
  - `pytspredit.plot`: adicionar `plot_ts` + `plot_series` (estilo auto-contido, retorna
    `Figure`), além dos já existentes `plot_ts_pred`/`plot_ts_pred_mv`; exportar no `__init__`.
  - `tspredit` (R): adicionar `grf_ts.R`/`grf_ts_pred.R`/`grf_series.R` (próprios), NAMESPACE,
    Rd. (Duplicação transitória com o daltoolbox — resolvida na Fase 3.)
  - Corrigir o docstring obsoleto do `pytspredit/plot.py` ("pydaltoolbox does not ship…").
  - Validar: ruff+pytest+import (Py); `R CMD INSTALL`/parity (R). **Paridade visual R↔Py.**
- **Fase 2 — migrar consumidores** para os donos (tspredit/pytspredit): exemplos do próprio
  daltoolbox, daltoolboxdp, notebooks do pydaltoolbox, reproduce (este só com OK do Eduardo).
- **Fase 3 — REMOVER do toolbox geral:** dropar `grf_ts`/`grf_ts_pred`/`grf_series` do
  `daltoolbox` (R: arquivos+NAMESPACE+Rd) e `plot_ts`/`plot_ts_pred`/`plot_series` do
  `pydaltoolbox.viz` (temporal.py + `viz/__init__` + `__init__` + `[viz]` doc). Bump de
  versão dos dois toolboxes. Validar tudo verde.
- **Fase 4 — enriquecer os exemplos Python** do pytspredit com os plots do R (alinhar ao
  ~70% do R): por família, adicionar `plot_ts`/`plot_ts_pred`/`plot_ts_pred_mv`; conversor
  verde. (Os exemplos R já têm; é trazer o Python ao nível do R.)

## Ritual
Branch por frente (`plot-ownership` no pytspredit e no tspredit; branches próprias nos
demais repos), exemplos committados, links do GitHub ao Eduardo, ff-merge só quando ele
pedir, `twine upload` só com o "pode fazer". Verde sempre (ruff+pytest+conversor; R CMD).
