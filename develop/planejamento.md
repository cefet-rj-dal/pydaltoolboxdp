# Planejamento — pydaltoolboxdp (Deep ++)

Documento vivo: contém somente o que falta. O concluído vai para `historico.md`; a
correspondência R↔Python fica em `mapeamento.md`; o protocolo de validação, em `parity.md`.

## Roadmap incremental

### 0.7.0 — corte arquitetural e maturidade

- zero `source_python`/`system.file("python", ...)` no R;
- remover todo `daltoolboxdp/inst/python` após substituição comprovada;
- exemplos Deep ++ ausentes de daltoolbox/tspredit/harbinger e presentes nos dois DP;
- matriz de paridade completa, auditoria de crédito e instalação limpa CPU-only;
- preparar beta, sem publicar até autorização explícita do Eduardo.

## Regras permanentes

- Bug em qualquer lado: localizar nos dois; pedir confirmação antes de corrigir o R.
- Toda classe: teoria + `References:` + `Porting:` com fidelidade evidenciada.
- Mesma entrada/split/seed nos dois lados; “aproximado” exige evidência.
- PyTorch roda em CPU por padrão; CUDA é teste opcional, nunca requisito do CI.
- Mudanças em repositórios irmãos atualizam a memória ativa deles no mesmo marco.
- Sem upload PyPI, push ou release sem autorização explícita.
