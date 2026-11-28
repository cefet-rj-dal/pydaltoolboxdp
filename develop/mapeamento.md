# Mapeamento R ↔ Python — Deep ++

Convenções: construtores S3 em `snake_case` tornam-se classes Python em `PascalCase`;
genéricos S3 tornam-se métodos; a implementação Python vive somente em `pydaltoolboxdp`.

Legenda: ✅ implementado · 🔜 próximo · ⬜ planejado · 🗑 redundante/remover.

## Autoencoders (`pydaltoolbox`)

| R `daltoolboxdp` | Python | Base | Status |
|---|---|---|---|
| `autoenc_e` | `AutoencE` | `AutoencBaseE` | ✅ 0.1.0 (ponte R e paridade) |
| `autoenc_ed` | `AutoencEd` | `AutoencBaseEd` | ✅ 0.1.0 (ponte R e paridade) |
| `autoenc_denoise_e/ed` | `AutoencDenoiseE/Ed` | `AutoencBaseE/Ed` | ✅ 0.1.0 (ponte R e paridade) |
| `autoenc_conv_e/ed` | `AutoencConvE/Ed` | `AutoencBaseE/Ed` | ✅ 0.1.0 (ponte R e paridade) |
| `autoenc_lstm_e/ed` | `AutoencLstmE/Ed` | `AutoencBaseE/Ed` | ✅ 0.2.0 local (ponte R e paridade) |
| `autoenc_stacked_e/ed` | `AutoencStackedE/Ed` | `AutoencBaseE/Ed` | ✅ 0.3.0 (ponte R e paridade) |
| `autoenc_variational_e/ed` | `AutoencVariationalE/Ed` | `AutoencBaseE/Ed` | ✅ 0.3.0 (ponte R e paridade) |
| `autoenc_adv_e/ed` | `AutoencAdvE/Ed` | `AutoencBaseE/Ed` | ✅ 0.3.0 (ponte R e paridade) |

## Classificação e regressão (`pydaltoolbox`)

| R | Python | Base | Status |
|---|---|---|---|
| `skcla_gb` | `SkclaGb` | `Classification` | ✅ 0.4.0 local (ponte R e paridade) |
| `torch_cla_mlp` | `TorchClaMlp` | `Classification` | ✅ 0.4.0 local (ponte R e paridade) |
| `torch_reg_mlp` | `TorchRegMlp` | `Regression` | ✅ 0.4.0 local (ponte R e paridade) |
| `skcla_knn` | `ClaKnn` (pydaltoolbox) | — | 🗑 removido do DP em 0.4.0 |
| `skcla_nb` | `ClaNb` (pydaltoolbox) | — | 🗑 removido do DP em 0.4.0 |
| `skcla_svc` | `ClaSvm` (pydaltoolbox) | — | 🗑 removido do DP em 0.4.0 |
| `skcla_rf` | `ClaRf` (pydaltoolbox) | — | 🗑 removido do DP em 0.4.0 |
| `skcla_mlp` | `ClaMlp` (pydaltoolbox) | — | 🗑 removido do DP em 0.4.0 |

## Séries temporais (`pytspredit`)

| R | Python | Base | Status |
|---|---|---|---|
| `torch_ts_mlp` | `TorchTsMlp` | `TsRegSW` | ✅ 0.5.0 local (ponte R e paridade) |
| `ts_conv1d` | `TsConv1D` | `TsRegSW` | ✅ 0.5.0 local (ponte R e paridade) |
| `ts_lstm` | `TsLstm` | `TsRegSW` | ✅ 0.5.0 local (ponte R e paridade) |

## Detecção de eventos (`pyharbinger`)

| R `harbinger` | Python Deep ++ | Base | Status |
|---|---|---|---|
| `han_autoencoder` | `HanAutoencoder` | `Harbinger` | ✅ 0.6.0 local (paridade exata) |
| `hcp_waypoint` | `HcpWaypoint` | `Harbinger` | ✅ 0.6.0 local (paridade exata) |

Os contratos/orquestrações genéricos permanecem no harbinger. Os modelos concretos e todos
os exemplos que usam um backend Deep ++ pertencem a daltoolboxdp/pydaltoolboxdp.
