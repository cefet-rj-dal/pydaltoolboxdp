Retomada do pydaltoolboxdp (C:\Git\dal\pydaltoolboxdp), camada **Deep ++**. Leia integralmente
`develop/{planejamento,historico,mapeamento,parity,deploy}.md` antes de agir.

Arquitetura travada: pydaltoolboxdp é a única implementação Python; daltoolboxdp importa as
classes via reticulate e perderá `inst/python` família a família, somente após paridade.
Exemplos concretos Deep ++ pertencem aos dois pacotes DP, não aos pacotes-base.

Estado: 0.5.0 está publicado no PyPI, com
`TorchTsMlp`, `TsConv1D` e `TsLstm` sobre `pytspredit.TsRegSW`, previsão one-step e
multi-step, referências congeladas, exemplos Python/R e pontes reticulate exatas a `1e-7`.
Os três backends embarcados foram removidos do daltoolboxdp; os exemplos Deep ++ saíram do
tspredit e foram consolidados nos dois pacotes DP.

Próximo corte: Harbinger Deep 0.6.0. Antes de qualquer edição, ler integralmente e depois
atualizar a memória ativa de `harbinger` e `pyharbinger`, pois os repositórios podem estar
em outro ciclo. Implementar `HanAutoencoder` e `HcpWaypoint` sobre `pyharbinger.Harbinger`
e transferir os 12 exemplos Deep ++ previstos no planejamento.

O planejamento contém apenas trabalho pendente; toda conclusão vai para o histórico. Não
fazer novas publicações, commits ou push sem autorização explícita do Eduardo.
