# Paridade Deep ++ — metodologia e matriz

## Protocolo

Cada família passa por três gates:

1. **Legado → classe nova:** enquanto o backend embarcado existir, executar a mesma entrada,
   arquitetura, pesos/seed, batches e épocas contra ele e contra pydaltoolboxdp.
2. **Python direto → R/reticulate:** chamar a mesma classe publicada pelos dois caminhos;
   diferenças aqui são defeitos de ponte/conversão.
3. **Resultado científico:** medir o invariante da família (representação, reconstrução,
   previsão ou eventos), não apenas ausência de erro.

CPU e float32 são a referência de paridade. CUDA é validada separadamente porque kernels e
reduções podem variar. Toda tolerância deve vir acompanhada de evidência.

## Invariantes

| Família | Invariante |
|---|---|
| encoder | matriz latente, forma e finitude |
| encoder-decoder | reconstrução e MSE por observação |
| classificação | ordem de classes, probabilidades e decisão |
| regressão | vetor previsto e métricas |
| TS | previsões one-step/multi-step |
| HanAutoencoder | resíduos, limiar e índices de anomalia |
| HcpWaypoint | resíduos, tau e índices de mudança |

## Matriz atual

| Componente | Gate 1 | Gate 2 | Gate 3 | Guarda |
|---|---|---|---|---|
| `autoenc_e` → `AutoencE` | exato (`1e-7`) | exato (`1e-7`) | forma, finitude e seed | `test_autoencoder.py`, `test_quality_dense.py` |
| `autoenc_ed` → `AutoencEd` | exato (`1e-7`) | exato (`1e-7`) | reconstrução, MSE e seed | `test_autoencoder.py`, `test_quality_dense.py` |
| `autoenc_denoise_e` → `AutoencDenoiseE` | exato (`1e-7`) | exato (`1e-7`) | latente limpo após treino ruidoso | `test_quality_denoise.py` |
| `autoenc_denoise_ed` → `AutoencDenoiseEd` | exato (`1e-7`) | exato (`1e-7`) | reconstrução limpa e seed | `test_quality_denoise.py` |

O backend denso foi apagado somente depois que os gates 1 e 2 ficaram verdes. A referência
legada permanece congelada em `develop/parity/dense_ref.json`, e a ponte é exercitada por
`daltoolboxdp/develop/parity/dense_reticulate.R`.
