Vamos implementar a 0.6.0 do pytspredit (c:\Git\dal\pytspredit). Leia primeiro a
memória [[pytspredit-port]] e develop/planejamento.md (§5.4, bloco 0.6.0) para o
contexto — 0.1.0→0.5.0 já estão publicados no PyPI e tudo foi ff-merged na main
(só a branch main existe).

ESCOPO 0.6.0 — "Benchmark (parte leve)", 0 dependência nova:
- Miniaturas de datasets embarcadas no pacote, extraídas dos .RData do R via
  Rscript: CSV para data.frame (ex.: NN5, SantaFe.D, EUNITE.Loads) e JSON para
  listas de séries/data.frames (ex.: m4, stocks, bioenergy), + um manifest.json
  com {name: {url, kind}} (as urls são as mesmas do repo cefet-rj-dal/tspredbench,
  guardadas como attr(x,"url") no R). datasets.<nome>() devolve a miniatura.
  tsd continua embarcado (sem url).
- TsPredBench: harness datasets × modelos (fit/predict/evaluate em grade).
- loadfulldata (download remoto) NÃO entra agora — é Ciclo 2 (0.9.0, extra [bench]).

RITUAL OBRIGATÓRIO (igual às releases anteriores):
- Branch por milestone (milestone-0.6.0-benchmark); exemplos sempre commitados;
  me passar os links do GitHub ao final para eu revisar; ff-merge na main só
  quando eu pedir; o twine upload eu autorizo/rodo.
- Paridade R vs Python a cada componente: R 4.5.0 em /c/R/R-4.5.0, pacote R em
  c:\Git\dal\tspredit (main já tem o fix do ts_norm_none). Conferir dims/colunas/
  valores das miniaturas contra o R; documentar tolerância onde o backend divergir.
- Convenção de documentação (travada na 0.5.0): TODA classe nova com References:
  (citações do @references do arquivo R) + Porting: (nome do arquivo R + nível
  exact/approximate/adapted + adaptações).
- Verde sempre: ruff check . + pytest + python tools/build_examples.py; um notebook
  de exemplo didático (notebooks/benchmark/…) + testes unitários; bump de versão
  em pyproject.toml e src/pytspredit/__init__.py; atualizar README e planejamento.md.

COMECE decidindo comigo as sub-decisões em aberto antes de codar:
(1) formato exato das miniaturas (CSV p/ data.frame, JSON p/ lista) e quantas
    linhas por miniatura; (2) quais datasets embarcar nesta release; (3) a API do
    TsPredBench (entrada datasets×modelos, saída tabela de métricas). Proponha e
    aguarde meu ok.

PRINCÍPIO ESTENDIDO — ++ & redução de dependência (2026-11-21): Eduardo estendeu a
iniciativa de back-implementation (ver memória [[back-implementation-initiative]])
para TODO o stack DAL, incluindo o pytspredit. A partir de agora, em paralelo ao
"não reimplementar — envolver libs canônicas": (1) incrementar métodos para "++"
quando fizer sentido — gate da Ação 1++ = prever a verdade tão bem ou melhor que a
referência, não mera paridade; (2) reduzir a pegada de deps back-implementando as
atípicas/niche nativamente (Python = releitura canônica / R = backport
100%-compatível; algoritmos-base compartilhados moram no pydaltoolbox). NÃO é regra
cega — caso a caso comigo; scikit-learn/scipy/etc. seguem embrulhados. O 0.6.0
(benchmark, 0 dep nova) não muda de escopo. TODO antes de escolher o PRÓXIMO marco:
auditar os imports deste pacote e montar a lista priorizada de candidatos ++ /
redução de dep numa seção do develop/planejamento.md.
