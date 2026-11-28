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
| `autoenc_conv_e` → `AutoencConvE` | bit a bit (`0.0`) | exato (`1e-7`) | latente, forma e seed | `test_quality_conv.py` |
| `autoenc_conv_ed` → `AutoencConvEd` | bit a bit (`0.0`) | exato (`1e-7`) | reconstrução, forma e seed | `test_quality_conv.py` |
| `autoenc_lstm_e` → `AutoencLstmE` | bit a bit (`0.0`) | exato (`1e-7`) | latente temporal, forma e seed | `test_quality_lstm.py` |
| `autoenc_lstm_ed` → `AutoencLstmEd` | bit a bit (`0.0`) | exato (`1e-7`) | reconstrução temporal e MSE | `test_quality_lstm.py` |
| `autoenc_stacked_e` → `AutoencStackedE` | bit a bit (`0.0`) | exato (`1e-7`) | latente hierárquico, forma e seed | `test_quality_stacked.py` |
| `autoenc_stacked_ed` → `AutoencStackedEd` | bit a bit (`0.0`) | exato (`1e-7`) | reconstrução reversa e MSE | `test_quality_stacked.py` |
| `autoenc_variational_e` → `AutoencVariationalE` | bit a bit (`0.0`) | exato (`1e-7`) | parâmetros latentes e seed | `test_quality_variational.py` |
| `autoenc_variational_ed` → `AutoencVariationalEd` | bit a bit (`0.0`) | exato (`1e-7`) | reconstrução estocástica, MSE e seed | `test_quality_variational.py` |
| `autoenc_adv_e` → `AutoencAdvE` | bit a bit (`0.0`) | exato (`1e-7`) | latente adversarial, forma e seed | `test_quality_adv.py` |
| `autoenc_adv_ed` → `AutoencAdvEd` | bit a bit (`0.0`) | exato (`1e-7`) | reconstrução adversarial e MSE | `test_quality_adv.py` |
| `skcla_gb` → `SkclaGb` | bit a bit (`0.0`) | exato (`1e-7`) | ordem de classes, probabilidades e decisão | `test_quality_supervised.py` |
| `torch_cla_mlp` → `TorchClaMlp` | bit a bit (`0.0`) | exato (`1e-7`) | loss, probabilidades, decisão e seed | `test_supervised.py`, `test_quality_supervised.py` |
| `torch_reg_mlp` → `TorchRegMlp` | bit a bit (`0.0`) | exato (`1e-7`) | loss, vetor previsto, métricas e seed | `test_supervised.py`, `test_quality_supervised.py` |
| `torch_ts_mlp` → `TorchTsMlp` | bit a bit (`0.0`) | exato (`1e-7`) | loss, one-step, multi-step e seed | `test_timeseries.py`, `test_quality_timeseries.py` |
| `ts_conv1d` → `TsConv1D` | exato (`1e-7`) | exato (`1e-7`) | loss, one-step, multi-step e seed | `test_timeseries.py`, `test_quality_timeseries.py` |
| `ts_lstm` → `TsLstm` | bit a bit (`0.0`) | exato (`1e-7`) | loss, one-step, multi-step e seed | `test_timeseries.py`, `test_quality_timeseries.py` |
| `han_autoencoder` → `HanAutoencoder` | exato (`1e-7`) | n/a | resíduos, limiar, eventos e seed | `test_events.py`, `test_quality_events.py` |
| `hcp_waypoint` → `HcpWaypoint` | exato (`1e-7`) | n/a | resíduos, tau, drift e seed | `test_events.py`, `test_quality_events.py` |

O backend denso foi apagado somente depois que os gates 1 e 2 ficaram verdes. A referência
legada permanece congelada em `develop/parity/dense_ref.json`, e a ponte é exercitada por
`daltoolboxdp/develop/parity/dense_reticulate.R`.

As referências das famílias seguintes estão em `develop/parity/denoise_ref.json` e
`develop/parity/conv_ref.json`, `develop/parity/lstm_ref.json`,
`develop/parity/stacked_ref.json`, `develop/parity/variational_ref.json` e
`develop/parity/adv_ref.json`; seus gates R ficam nos scripts homônimos sob
`daltoolboxdp/develop/parity`.

A referência da classificação/regressão está em `develop/parity/supervised_ref.json`;
o gate R conjunto está em `daltoolboxdp/develop/parity/supervised_reticulate.R`.

A referência das séries temporais está em `develop/parity/timeseries_ref.json`; o gate R
conjunto está em `daltoolboxdp/develop/parity/timeseries_reticulate.R`. A diferença máxima
do Conv1D legado foi um ULP (`5.96e-8`), dentro da tolerância exata adotada de `1e-7`.

A referência dos detectores Deep está em `develop/parity/events_ref.json`; o gate conjunto
R↔Python está em `develop/parity/events_reticulate.R`. Como os contratos R permanecem no
harbinger e usam os autoencoders públicos do DP, o gate compara diretamente as duas
orquestrações sobre o mesmo backend: diferenças máximas de `2.7e-15` (anomalias) e
`1.5e-14` (Waypoint), com eventos e `tau` idênticos.
