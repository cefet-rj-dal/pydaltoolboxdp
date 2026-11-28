# Planejamento — pydaltoolboxdp (Deep ++)

Documento vivo: contém somente o que falta. O concluído vai para `historico.md`; a
correspondência R↔Python fica em `mapeamento.md`; o protocolo de validação, em `parity.md`.

## Roadmap incremental

### 0.4.0 — classificação e regressão

- `SkclaGb`, `TorchClaMlp`, `TorchRegMlp`;
- manter `skcla_gb` no R porque não existe em pydaltoolbox;
- remover de daltoolboxdp os redundantes `skcla_knn/nb/svc/rf/mlp`, seus backends e
  exemplos, usando as classes de daltoolbox/pydaltoolbox;
- remover os scripts `skimb_*` órfãos, se a auditoria confirmar ausência de API pública.

### 0.5.0 — séries temporais

- `TorchTsMlp`, `TsConv1D`, `TsLstm`, todos sobre `pytspredit.TsRegSW`;
- previsão one-step e multi-step;
- transferir de tspredit os 3 exemplos de prediction e 3 de multivariate, consolidando
  duplicatas já existentes no daltoolboxdp;
- atualizar os plots para a propriedade atual de tspredit/pytspredit.

### 0.6.0 — harbinger deep

- `HanAutoencoder` e `HcpWaypoint`, especializando `pyharbinger.Harbinger`;
- ler integralmente a memória ativa de harbinger/pyharbinger antes de qualquer edição;
- transferir 12 exemplos Deep ++ hoje no harbinger (2 HanrMl deep, 7 autoencoder, 3
  Waypoint) e criar seus pares Python;
- registrar toda intervenção nos `develop/{planejamento,historico,mapeamento,parity,
  nextprompt}.md` do pyharbinger para não conflitar com seu ciclo em andamento.

### 0.7.0 — corte arquitetural e maturidade

- zero `source_python`/`system.file("python", ...)` no R;
- remover todo `daltoolboxdp/inst/python` após substituição comprovada;
- exemplos Deep ++ ausentes de daltoolbox/tspredit/harbinger e presentes nos dois DP;
- matriz de paridade completa, auditoria de crédito e instalação limpa CPU-only;
- preparar beta, sem publicar até autorização explícita do Eduardo.

## Regras permanentes

- Bug em qualquer lado: localizar nos dois; pedir confirmação antes de corrigir o R.
- Toda classe: teoria + `References:` + `Porting:` com fidelidade evidenciada.
- Mesma entrada/split/seed nos dois lados; “aproximado” exige evidência.
- PyTorch roda em CPU por padrão; CUDA é teste opcional, nunca requisito do CI.
- Mudanças em repositórios irmãos atualizam a memória ativa deles no mesmo marco.
- Sem upload PyPI, push ou release sem autorização explícita.
