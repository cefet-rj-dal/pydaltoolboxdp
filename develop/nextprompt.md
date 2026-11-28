Retomada do pydaltoolboxdp (C:\Git\dal\pydaltoolboxdp), camada **Deep ++**. Leia integralmente
`develop/{planejamento,historico,mapeamento,parity,deploy}.md` antes de agir.

Arquitetura travada: pydaltoolboxdp é a única implementação Python; daltoolboxdp importa as
classes via reticulate e perderá `inst/python` família a família, somente após paridade.
Exemplos concretos Deep ++ pertencem aos dois pacotes DP, não aos pacotes-base.

Estado: 0.5.0 está publicado no PyPI. O corte 0.6.0 está concluído localmente com
`HanAutoencoder` e `HcpWaypoint` sobre `pyharbinger.Harbinger`, referências congeladas,
gate R↔Python exato a `1e-7` e pares Python executados para as 12 configurações Deep.
Os exemplos R foram transferidos do harbinger para daltoolboxdp sem remover os contratos
genéricos do pacote-base. As memórias de harbinger/pyharbinger registram a nova propriedade.

Próximo corte: maturidade 0.7.0 — auditoria de zero backend embarcado, cobertura integral
dos exemplos Deep ++ nos dois DP, matriz de paridade, crédito e instalação CPU limpa.

O planejamento contém apenas trabalho pendente; toda conclusão vai para o histórico. Não
fazer novas publicações, commits ou push sem autorização explícita do Eduardo.
