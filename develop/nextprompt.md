Retomada do pydaltoolboxdp (C:\Git\dal\pydaltoolboxdp), camada **Deep ++**. Leia integralmente
`develop/{planejamento,historico,mapeamento,parity,deploy}.md` antes de agir.

Arquitetura travada: pydaltoolboxdp é a única implementação Python; daltoolboxdp importa as
classes via reticulate e perderá `inst/python` família a família, somente após paridade.
Exemplos concretos Deep ++ pertencem aos dois pacotes DP, não aos pacotes-base.

Estado: 0.2.0 publicado no PyPI e reinstalado localmente. `AutoencE/Ed`, `AutoencDenoiseE/Ed`,
`AutoencConvE/Ed` e `AutoencLstmE/Ed` estão migrados, com exemplos, referências congeladas e
pontes R. Os gates deram paridade exata a `1e-7` (Conv1D e LSTM foram bit a bit no gate legado),
e seus quatro backends embarcados foram removidos. Próximo corte: `AutoencStackedE/Ed` (0.3.0), repetindo legado↔novo,
Python↔R, resultado científico, exemplos e remoção do backend somente ao final. O `R CMD check` atual tem
um warning preexistente de `.Rd` das famílias ainda não migradas. Não tocar harbinger/pyharbinger
sem ler e atualizar sua memória ativa; eles estão no meio de outro ciclo.

Os exemplos Conv1D e LSTM R/Python estão alinhados em arquitetura, seed, épocas, batch e
métricas. O planejamento contém apenas
trabalho pendente; toda conclusão vai para o histórico. Não fazer novas publicações, commits ou push sem
autorização explícita do Eduardo.
