Retomada do pytspredit (c:\Git\dal\pytspredit). Leia primeiro a memória [[pytspredit-port]],
develop/planejamento.md (só o que falta), develop/historico.md (a história), develop/mapeamento.md
e develop/parity.md. **O beta está PUBLICADO e em soak.**

ESTADO ATUAL (2026-11-25)
- **pytspredit 0.13.0 — RELEASE BETA — PUBLICADO no PyPI** (https://pypi.org/project/pytspredit/0.13.0/),
  na main (HEAD `ad5da66`). Classificador `Development Status :: 4 - Beta`. Fecha a maturidade
  **M1–M6 ✓** + a **tarefa extra pré-beta (migração dos plots de TS + auditoria de paridade dos
  exemplos) ✓**. **Ciclo 1 + Ciclo 2 completos** — todo o `tspredit` portado, **0 dep nova**.
- Outras releases vivas: **pydaltoolbox 0.25.0** (PyPI, sem plots de TS) · **tspredit 2.0.757**
  e **daltoolbox 1.4.717** nas mains R (CRAN = Eduardo faz).
- `main` é a única branch de código nos dois repos. Exemplos: 70/104 notebooks plotam, varredura
  sem erro (fora 6 torch pré-existentes). 261 testes + ruff verdes.

O QUE FALTA (ver planejamento.md)
1. **Soak do beta ~1 MÊS** (até ~2026-12-25; cadência do pydaltoolbox) coletando feedback.
2. **Promover a 1.0.0 estável** — congela a API, sai de beta. **A partir da 1.0.0 o pytspredit
   também adota numeração Boeing** (1.0.707, 1.0.717, …).
3. **Endurecimento de paridade** (na passada da 1.0.0): re-evidenciar sob seed os 2 vereditos
   benignos ainda asseridos — RNG do augmentation e RNG das folds do tuning (parity.md §8).

PENDÊNCIA EXTERNA (Eduardo faz)
- Submissão ao CRAN: `tspredit 2.0.757` (CRAN-ready: 0E/0W/1 NOTE benigno) e `daltoolbox 1.4.717`.

DIFERIDO / EM ABERTO
- Migrar `daltoolboxdp` e `reproduce` para os plots de TS realocados (fora do núcleo, p/ depois).
- Plots dos tutoriais 05/07/09/10 (comparação/tuning) — adicionar se o Eduardo quiser paridade aí.

RITUAL (igual às releases anteriores)
- Branch por milestone; exemplos sempre committados; links do GitHub ao final; ff-merge só quando
  o Eduardo pedir; **o `twine upload` o Eduardo autoriza** (o build + `twine check` rodam normal).
- Verde sempre: `ruff check .` + `pytest` + `python notebooks/conversor.py` (UM argumento: família
  ou arquivo).
- Paridade committada (`develop/parity/` + `tests/test_quality_*.py`); R 4.5.0 em /c/R/R-4.5.0;
  pacote R em c:\Git\dal\tspredit. Convenção de doc: Teoria + `References:` + `Porting:` em toda
  classe de algoritmo. Exemplos do roxygen (R) têm que rodar RÁPIDO (ordens explícitas, nunca CV).

GOTCHAS DESTA MÁQUINA
- `library(tspredit)`/`pkgload::load_all` dão SEGFAULT intermitente (ABI/BLAS nativo, pré-existente,
  NÃO é o nosso código) → validar R via `R CMD INSTALL` + scripts standalone.
- `roxygen2` 8.1.1 → `roxygenise()` regenera NAMESPACE/Rd ok. `TsNormDiff` perde 1 coluna (em
  exemplos com diff, `input_size < sw-1`). ARIMA++ (`auto="cv"`) é lento (~15s) — fora de exemplos
  rápidos; benchmark em `develop/parity/autoarima_benchmark.py`.
- Depois de mover funções entre pacotes, grep o fonte R por `oldpkg::fn(` (chamadas com namespace
  explícito não resolvem por masking e quebram `R CMD check`). O conversor R knita TODOS os .Rmd e
  reporta `fail=0` mesmo com erro de chunk (error=TRUE) → grep `examples/**/*.md` por `^## Error`.

NÃO HÁ TAREFA ATIVA: o beta está no ar, em soak. Só retome aqui para (a) promover a 1.0.0 após o
mês, (b) o endurecimento de paridade, ou (c) um fix que surja do feedback do beta.
(Obs.: o trabalho do pyharbinger segue em OUTRA sessão/projeto — não é aqui.)
