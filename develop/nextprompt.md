Retomada do pydaltoolboxdp (C:\Git\dal\pydaltoolboxdp), camada **Deep ++**. Leia integralmente
`develop/{planejamento,historico,mapeamento,parity,deploy}.md` antes de agir.

Arquitetura travada: pydaltoolboxdp é a única implementação Python; daltoolboxdp importa as
classes via reticulate e perderá `inst/python` família a família, somente após paridade.
Exemplos concretos Deep ++ pertencem aos dois pacotes DP, não aos pacotes-base.

Estado: 0.3.1 publicado no PyPI e reinstalado sem cache a partir da distribuição pública.
As 14 classes das sete famílias de autoencoders estão migradas, com exemplos, referências
congeladas e pontes R. Os gates deram paridade exata a `1e-7` e os sete backends embarcados
foram removidos. Próximo corte: classificação e regressão 0.4.0, começando por `SkclaGb`,
`TorchClaMlp` e `TorchRegMlp`. Não tocar harbinger/pyharbinger
sem ler e atualizar sua memória ativa; eles estão no meio de outro ciclo.

Todos os exemplos de autoencoder R/Python estão alinhados em arquitetura, seed, épocas, batch e
métricas. O planejamento contém apenas
trabalho pendente; toda conclusão vai para o histórico. Não fazer novas publicações, commits ou push sem
autorização explícita do Eduardo.
