Retomada do pydaltoolboxdp (C:\Git\dal\pydaltoolboxdp), camada **Deep ++**. Leia integralmente
`develop/{planejamento,historico,mapeamento,parity,deploy}.md` antes de agir.

Arquitetura travada: pydaltoolboxdp é a única implementação Python; daltoolboxdp importa as
classes via reticulate e perderá `inst/python` família a família, somente após paridade.
Exemplos concretos Deep ++ pertencem aos dois pacotes DP, não aos pacotes-base.

Estado: 0.4.0 está concluído e validado apenas localmente, sem publicação. As três classes
tabulares (`SkclaGb`, `TorchClaMlp`, `TorchRegMlp`) têm exemplos, referências congeladas e
pontes R; os gates foram bit a bit no legado e exatos a `1e-7` via reticulate. Os cinco
classificadores redundantes e os scripts `skimb_*` órfãos foram removidos do daltoolboxdp.
Próximo corte: séries temporais 0.5.0, começando por `TorchTsMlp`, `TsConv1D` e `TsLstm`
sobre `pytspredit.TsRegSW`, incluindo one-step/multi-step e consolidação dos exemplos.
Não tocar harbinger/pyharbinger sem ler e atualizar sua memória ativa; eles estão no meio
de outro ciclo.

O `R CMD check` atual tem apenas o warning preexistente de assinaturas `.Rd` das três
famílias de séries temporais; corrigir junto da migração 0.5.0. O planejamento contém apenas
trabalho pendente; toda conclusão vai para o histórico. Não fazer novas publicações,
commits ou push sem autorização explícita do Eduardo.
