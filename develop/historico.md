# Histórico — pydaltoolboxdp

## Arquitetura consolidada

- `pydaltoolboxdp` é a única implementação Python da camada Deep ++; `daltoolboxdp`
  preserva adaptadores S3 finos via `reticulate` e perde os backends embarcados somente
  depois dos gates de cada família.
- Deep ++ especializa `pydaltoolbox` para modelos tabulares, `pytspredit` para séries
  temporais e `pyharbinger` para eventos.
- Exemplos de classes concretas Deep ++ pertencem aos dois pacotes DP, não aos pacotes-base.

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

## Denoising — 0.1.0 (2026-11-27)

- Implementados `AutoencDenoiseE/Ed` com ruído gaussiano nas entradas de treino e alvos limpos.
- Comprovada paridade exata a `1e-7` contra o backend embarcado e pela ponte R/reticulate.
- Congelada a saída legada em `develop/parity/denoise_ref.json` e adicionada guarda ao pytest.
- Migrados os dois wrappers R e removido `inst/python/autoenc_denoise.py` após os gates verdes.
- Criados e executados os exemplos Python 02 e 12; os exemplos R equivalentes permanecem no DP.
- Corrigidas as assinaturas `.Rd` apenas desta família, reduzindo a dívida sem misturar escopos.

## Conv1D — 0.1.0 (2026-11-27)

- Implementados `AutoencConvE/Ed` com reshape interno para um canal, duas convoluções no
  encoder, bottleneck linear e decoder com convoluções transpostas e saída sigmoide.
- Comprovada paridade bit a bit contra o backend legado e paridade exata a `1e-7` pela
  ponte R/reticulate; a referência ficou congelada em `develop/parity/conv_ref.json`.
- Migrados os wrappers R e removido `inst/python/autoenc_conv.py` somente após os gates.
- Criados e executados os exemplos Python 03 e 13, cobrindo latente e reconstrução.
- Corrigidas as assinaturas `.Rd` somente da família Conv1D.

## Publicação inicial — 0.1.0 (2026-11-27)

- Alinhados os exemplos Conv1D R/Python em arquitetura, seed, épocas, batch e métricas;
  os quatro exemplos foram executados com sucesso.
- Publicados wheel e sdist no PyPI após `twine check` e instalados novamente sem cache.
- Confirmados, a partir de `site-packages`, versão `0.1.0`, encode `(6, 2)` e reconstrução
  `(6, 4)`; os exemplos R também rodaram usando a distribuição pública instalada.

## LSTM e publicação — 0.2.0 (2026-11-27)

- Implementados `AutoencLstmE/Ed` com reshape configurável de sequência, encoder LSTM,
  projeção latente e decoder LSTM, mantendo suporte a profundidade e dropout recorrentes.
- O gate legado↔novo foi bit a bit para loss, latente e reconstrução; o gate Python↔R
  foi exato a `1e-7` para os dois adaptadores.
- Congelada a referência em `develop/parity/lstm_ref.json`, adicionados testes de contrato,
  seed, reshape e qualidade, e removido `inst/python/autoenc_lstm.py` após os gates verdes.
- Alinhados e executados os exemplos 04/14 R e Python com arquitetura, seed, épocas, batch
  e métricas idênticos.
- Atualizada a versão para `0.2.0`; sdist e wheel passaram por `twine check` e foram
  publicados no PyPI com autorização explícita.
- A wheel pública foi reinstalada sem cache e o smoke test confirmou versão 0.2.0 e
  reconstrução LSTM `(6, 5)` a partir de `site-packages`.
- Como a descrição publicada da 0.2.0 ainda fixava o comando de instalação em 0.1.0,
  a instrução foi tornada não versionada e a correção documental 0.2.1 foi publicada e
  reinstalada a partir do índice público.

## Stacked — 0.3.0 parcial (2026-11-27)

- Implementados `AutoencStackedE/Ed` com treinamento guloso por estágio, larguras latentes
  configuráveis, arquiteturas densas compartilhadas ou específicas e decodificação reversa.
- Os gates legado↔novo foram bit a bit para loss, latente e reconstrução; os adaptadores
  Python↔R foram exatos a `1e-7` para encode e encode-decode.
- Congelada a referência em `develop/parity/stacked_ref.json`, adicionados testes de contrato,
  seed, configuração por estágio e qualidade, e removido `inst/python/autoenc_stacked.py`.
- Alinhados e executados os exemplos R/Python 05/15 com seed, arquitetura, épocas, batch e
  métricas idênticos. Não houve nova publicação.
- `ruff` passou, o pytest terminou com 23 testes, sdist/wheel passaram por `twine check` e
  o smoke test da wheel local confirmou reconstrução Stacked `(6, 4)`.

## VAE, AAE e fechamento dos autoencoders — 0.3.0 (2026-11-27)

- Implementados `AutoencVariationalE/Ed` e `AutoencAdvE/Ed`, fechando as 14 classes das
  sete famílias de autoencoders Deep ++.
- Os gates legado↔novo foram bit a bit; os quatro adaptadores R foram exatos a `1e-7`,
  inclusive com reinício explícito de seed nas transformações estocásticas do VAE.
- Congeladas referências para VAE e AAE, adicionadas guardas de contrato, seed e qualidade,
  e removidos os dois backends embarcados somente após os gates verdes.
- Alinhados e executados os exemplos R/Python 06/16 e 07/17 com arquitetura, seed, épocas,
  batch e métricas MSE/R² idênticos.
- Publicados wheel e sdist 0.3.0 no PyPI com autorização explícita; a wheel pública foi
  reinstalada sem cache e os smoke tests confirmaram VAE e AAE com saída `(6, 4)`.
- Como a descrição empacotada da 0.3.0 ainda marcava 0.3.x como local, a correção documental
  0.3.1 foi publicada e reinstalada da wheel pública, mantendo os mesmos smoke tests verdes.

## Classificação e regressão — 0.4.0 (2026-11-27)

- Implementados `SkclaGb`, `TorchClaMlp` e `TorchRegMlp` sobre os contratos
  `Classification`/`Regression` de pydaltoolbox, com CPU por padrão, seed opcional,
  probabilidades ordenadas e históricos de treino/validação.
- Os gates legado→novo foram bit a bit para probabilidades, decisões, loss e regressão;
  os três adaptadores R→Python foram exatos a `1e-7`.
- Congelada a referência em `develop/parity/supervised_ref.json`, adicionadas guardas de
  contrato, reprodutibilidade, early stopping e qualidade, e removidos os três backends
  embarcados somente após os gates verdes.
- Criados, executados e renderizados exemplos Python para Gradient Boosting, MLP de
  classificação e MLP de regressão; os exemplos R concretos permanecem no daltoolboxdp.
- Removidos de daltoolboxdp os APIs redundantes `skcla_knn/nb/svc/rf/mlp`, seus backends,
  documentação e exemplos; os equivalentes vivem em daltoolbox/pydaltoolbox.
- A auditoria confirmou que `skimb_smote`, `skimb_smote_tomek` e `skimb_tomek_links` não
  tinham API pública nem consumidores; os scripts órfãos foram removidos, assim como o
  helper `autoenc_common.py` já sem consumidores.
- `ruff` passou, o pytest terminou com 38 testes, sdist/wheel 0.4.0 passaram por
  `twine check`, e `R CMD check` concluiu com apenas o warning preexistente das assinaturas
  `.Rd` das três famílias de séries temporais previstas para 0.5.0.
- Wheel e sdist 0.4.0 foram publicados no PyPI com autorização explícita; a wheel pública
  foi reinstalada sem cache em `site-packages`, o smoke test confirmou `SkclaGb` e o gate
  R/reticulate permaneceu exato a `1e-7` sem usar a árvore-fonte.
