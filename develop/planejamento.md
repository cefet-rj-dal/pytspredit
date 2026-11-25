# Planejamento — o que falta no pytspredit

> Documento vivo. **Contém só o que resta fazer.** A história do que já foi concluído está
> em `historico.md`; a correspondência de classes R↔Python em `mapeamento.md`; a metodologia
> e a matriz de paridade em `parity.md`.

Última atualização: 2026-11-25

---

## Onde estamos (resumo de 1 linha)

Todo o `tspredit` portado (Ciclo 1 + Ciclo 2, **sem dep nova**); maturidade **M1–M6 ✓** e a
**tarefa extra pré-beta (migração dos plots de TS + exemplos) ✓**; últimas releases no PyPI:
**pytspredit 0.12.1** e **pydaltoolbox 0.25.0**. Falta só **fechar o beta e promover a 1.0.0**.
Detalhe do caminho até aqui: `historico.md`.

---

## O que falta — roadmap até a 1.0.0

### 1. ✅ Release BETA — `0.13.0`
Decisão do Eduardo (2026-11-24): **não** vamos para `1.0.0b1`; batizamos de **beta a próxima
`0.x.0`**. **Eduardo (2026-11-25): a 0.13.0 é o beta — publicada no PyPI.** Classificador
`Development Status :: 4 - Beta`. Pré-requisitos (M1–M6 + tarefa extra dos plots) todos ✓.
- **Em curso:** rodar **~1 mês** em beta (como no pydaltoolbox), coletar feedback de uso, antes
  de promover a 1.0.0.

### 2. ⬜ Release `1.0.0` — estável
Depois da semana de beta sem problemas: congela a API, sai de beta. **A partir da 1.0.0 o
pytspredit também adota numeração Boeing** (1.0.707, 1.0.717, …), como o `tspredit` R.

### 3. ⬜ Endurecimento de paridade (na passada da 1.0.0)
Re-evidenciar sob **seed/regime-concordante** os poucos vereditos "benignos" que ainda são
*asseridos* e não *re-rodados* (ver `parity.md` §8): **RNG do augmentation** (jitter/awareness/
awaresmooth) e **RNG das folds do tuning**. As linhas estocásticas dos regressores já citam
seed/regime; é fechar essas duas.

---

## Pendências externas (Eduardo faz)
- **Submissão ao CRAN:** `tspredit 2.0.757` (CRAN-ready: 0E/0W/1 NOTE benigno) e
  `daltoolbox 1.4.717`. Os pacotes R o Eduardo publica.

## Diferido (combinado: depois do núcleo)
- **Migrar `daltoolboxdp` e `reproduce`** para os plots de TS realocados (hoje donos em
  `(py)tspredit`). A migração de responsabilidade foi feita **só no núcleo**
  (daltoolbox/pydaltoolbox/tspredit/pytspredit); esses dois downstream ficaram para depois.

## Em aberto (decidir com o Eduardo)
- **Plots dos tutoriais 05/07/09/10** (notebooks de comparação/tuning) — hoje sem plot; os
  equivalentes do R podem plotar. Adicionar se o Eduardo quiser paridade também aí.

---

## Princípios ainda em vigor (guiam o que falta)

Estes continuam valendo para qualquer trabalho novo (o detalhe e o histórico de cada decisão
estão em `historico.md`):

- **Convenção de docstring:** toda classe de algoritmo traz **Teoria + `References:`
  (paper original) + `Porting:`** (`Mirrors <arquivo_r>.R`; fidelidade exact/approximate/
  adapted). Sem frases proibidas ("native implementation"/"no dependency"/alfinetadas) —
  redução de dep vai na **mensagem de commit**, não na doc.
- **Back-implementation (Python = releitura canônica, R = backport):** trocar dep de terceiros
  por implementação nativa **só** em casos atípicos/nicho (lib falta/não mantida/atrito de
  build/impede paridade). Libs canônicas (`sklearn`/`scipy`/`numpy`/`pandas`/`statsmodels`)
  **seguem embrulhadas**. **Gate ++:** toda back-impl ou melhoria algorítmica tem de **medir o
  gap vs. a referência** e, se for "++", **bater a baseline** com evidência. Decidido **caso a
  caso** com o Eduardo, nunca por automação.
- **Paridade a cada passo:** rodar R e Python sobre o mesmo input congelado, comparar o
  **invariante da família** (não a representação), e **travar em `tests/test_quality_*.py` +
  `develop/parity/<fam>.R`**. **Bug em qualquer lado corrige-se nos dois repos.** Reference
  não é verdade: paridade é concordância **depois dos dois certos**.
- **Como trabalhamos:** incremental, verde sempre (`ruff` + `pytest` + build dos exemplos);
  **branch por milestone** → push → Eduardo revisa no GitHub → ff-merge (sem `gh` aqui); nada
  na `main` sem o Eduardo pedir; cada upload ao PyPI exige o **"pode fazer"** dele.
