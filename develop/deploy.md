# Deploy — pydaltoolboxdp

Nenhuma publicação ocorre sem autorização explícita do Eduardo.

Checklist por release:

1. atualizar versão em `pyproject.toml` e `src/pydaltoolboxdp/__init__.py`;
2. `ruff check .`;
3. `pytest` em ambiente CPU limpo;
4. executar os gates de paridade da release;
5. gerar e verificar exemplos;
6. `python -m build` e `twine check dist/*`;
7. testar instalação da wheel em venv novo;
8. somente após autorização: upload e verificação do endpoint versionado do PyPI.

O pacote R compatível deve declarar uma versão mínima de pydaltoolboxdp. A matriz de
compatibilidade R↔Python será registrada no README antes do primeiro release público.
