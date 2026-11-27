# Histórico — pydaltoolboxdp

## Fundação Deep ++ — 0.1.0 validado localmente (2026-11-27)

- Definida a arquitetura: Python implementado uma única vez em pydaltoolboxdp; o pacote R
  passa a ser adaptador `reticulate`, sem backend Python duplicado.
- `dp` redefinido como **Deep ++**, adequado aos dois idiomas e às especializações de
  pydaltoolbox, pytspredit e pyharbinger.
- Inventariados 25 construtores públicos do daltoolboxdp, os backends órfãos, 6 exemplos
  deep no tspredit e 12 no harbinger.
- Criado o pacote Python e a infraestrutura inicial de treino/validação/stopping PyTorch.
- Implementados `AutoencE` e `AutoencEd`, com seed opcional, CPU por padrão, histórico de
  loss e override de épocas para consumidores adaptativos.
- Criado ambiente isolado para contornar a instalação global quebrada de torch/Python 3.14.
- Congelada uma referência do backend legado e comprovada paridade exata (tolerância `1e-7`)
  de loss, representação latente e reconstrução para `AutoencE/AutoencEd`.
- Migrados `daltoolboxdp::autoenc_e/autoenc_ed` para `pydaltoolboxdp` via helper único de
  importação `reticulate`; o backend duplicado `inst/python/autoenc.py` foi então removido.
- Gerados e executados os dois exemplos Python; `ruff` passou e `pytest` terminou com 7 testes.
- O sdist e a wheel passaram por `twine check`; não houve publicação.
- O pacote R foi construído e passou instalação, carga, exemplos e integração. O `R CMD check`
  terminou com um warning preexistente de assinaturas `.Rd` desatualizadas nas famílias ainda
  não migradas; ele será eliminado junto da documentação de cada família.
- Definido Python `>=3.10`, compatível com as anotações de tipo realmente usadas pelo código.

## Denoising — 0.2.0 parcial (2026-11-27)

- Implementados `AutoencDenoiseE/Ed` com ruído gaussiano nas entradas de treino e alvos limpos.
- Comprovada paridade exata a `1e-7` contra o backend embarcado e pela ponte R/reticulate.
- Congelada a saída legada em `develop/parity/denoise_ref.json` e adicionada guarda ao pytest.
- Migrados os dois wrappers R e removido `inst/python/autoenc_denoise.py` após os gates verdes.
- Criados e executados os exemplos Python 02 e 12; os exemplos R equivalentes permanecem no DP.
- Corrigidas as assinaturas `.Rd` apenas desta família, reduzindo a dívida sem misturar escopos.
