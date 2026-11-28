# Deploy — pydaltoolboxdp

Nenhuma publicação ocorre sem autorização explícita do Eduardo.

## Publicações

- `0.1.0` — publicada no PyPI em 2026-11-27 com autorização explícita; wheel e sdist
  passaram por `twine check`, instalação sem cache e smoke test local.
- `0.2.0` — publicada no PyPI em 2026-11-27 com autorização explícita; wheel e sdist
  passaram por `twine check`, reinstalação pública sem cache e smoke test LSTM local.
- `0.2.1` — correção da instrução de instalação exibida no PyPI, sem mudança de API;
  publicada e reinstalada sem cache em 2026-11-27.
- `0.3.0` — conclui as 14 classes de autoencoder; wheel e sdist passaram por `twine check`,
  foram publicados com autorização explícita e a wheel pública foi reinstalada sem cache
  com smoke tests VAE e AAE em 2026-11-27.
- `0.3.1` — corrige na descrição publicada a marcação residual de 0.3.x como local; wheel
  e sdist foram publicados e a wheel pública foi reinstalada sem cache em 2026-11-27.
- `0.4.0` — adiciona classificação/regressão tabulares e conclui a remoção dos equivalentes
  redundantes no pacote R; wheel e sdist passaram por `twine check`, foram publicados com
  autorização explícita e a wheel pública foi reinstalada sem cache. O gate R/reticulate
  contra `site-packages` permaneceu exato a `1e-7` em 2026-11-27.
- `0.5.0` — adiciona `TorchTsMlp`, `TsConv1D` e `TsLstm`; wheel e sdist passaram por
  `twine check`, foram publicados com autorização explícita e a wheel pública foi instalada
  sem cache em venv isolado. Os três modelos passaram em smoke tests de treino e previsão;
  o pacote R instalado do GitHub usou essa distribuição pública via reticulate em 2026-11-28.

Checklist por release:

1. atualizar versão em `pyproject.toml` e `src/pydaltoolboxdp/__init__.py`;
2. `ruff check .`;
3. `pytest` em ambiente CPU limpo;
4. executar os gates de paridade da release;
5. gerar e verificar exemplos;
6. `python -m build` e `twine check dist/*`;
7. testar instalação da wheel em venv novo;
8. somente após autorização: upload e verificação do endpoint versionado do PyPI.

## Preparado localmente

- `0.6.0` — adiciona `HanAutoencoder` e `HcpWaypoint`, com referências congeladas, gate
  R↔Python e exemplos das 12 configurações Deep. Ainda precisa dos gates finais de build,
  instalação isolada e pacote R antes de qualquer pedido de publicação.

O pacote R compatível deve declarar uma versão mínima de pydaltoolboxdp. A matriz de
compatibilidade R↔Python será registrada no README antes do primeiro release público.
