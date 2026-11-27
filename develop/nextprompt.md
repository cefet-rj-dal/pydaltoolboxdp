Retomada do pydaltoolboxdp (C:\Git\dal\pydaltoolboxdp), camada **Deep ++**. Leia integralmente
`develop/{planejamento,historico,mapeamento,parity,deploy}.md` antes de agir.

Arquitetura travada: pydaltoolboxdp é a única implementação Python; daltoolboxdp importa as
classes via reticulate e perderá `inst/python` família a família, somente após paridade.
Exemplos concretos Deep ++ pertencem aos dois pacotes DP, não aos pacotes-base.

Estado: 0.1.0 validado localmente. `AutoencE/AutoencEd`, exemplos, referência congelada e ponte
R estão prontos; os dois gates deram paridade exata a `1e-7`, e `inst/python/autoenc.py` foi
removido. `AutoencDenoiseE/Ed` também estão migrados, com exemplos e paridade exata; o backend
`autoenc_denoise.py` foi removido. Próximo corte: `AutoencConvE/Ed` (0.2.0), repetindo legado↔novo, Python↔R,
resultado científico, exemplos e remoção do backend somente ao final. O `R CMD check` atual tem
um warning preexistente de `.Rd` das famílias ainda não migradas. Não tocar harbinger/pyharbinger
sem ler e atualizar sua memória ativa; eles estão no meio de outro ciclo.

Não publicar, commitar ou fazer push sem autorização explícita do Eduardo.
