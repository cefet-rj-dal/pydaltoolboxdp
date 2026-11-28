# pydaltoolboxdp: Deep ++ Extensions for `pydaltoolbox`

`pydaltoolboxdp` is the Python implementation of the DAL ecosystem's **Deep ++**
layer. It provides deep-learning and advanced models that specialize the contracts
defined by `pydaltoolbox`, `pytspredit`, and `pyharbinger`.

The sibling R package, `daltoolboxdp`, is a thin `reticulate` interface to these
classes. Python algorithms live here once; they are not duplicated inside the R
package.

The implemented vertical slices contain all seven encoder and encoder-decoder
autoencoder families, tabular classification and regression, and time-series
forecasting:

- `AutoencE`: transforms observations into their latent representation;
- `AutoencEd`: reconstructs observations after encoding.
- `AutoencDenoiseE`: learns robust latent representations from corrupted inputs;
- `AutoencDenoiseEd`: reconstructs clean observations after denoising training.
- `AutoencConvE`: learns compact representations of local patterns with Conv1D layers;
- `AutoencConvEd`: reconstructs windows with a convolutional encoder-decoder.
- `AutoencLstmE`: learns recurrent representations of temporal windows;
- `AutoencLstmEd`: reconstructs windows with a recurrent encoder-decoder.
- `AutoencStackedE`: learns hierarchical representations through greedy stages;
- `AutoencStackedEd`: reconstructs observations through the reversed decoder stack.
- `AutoencVariationalE/Ed`: learns a probabilistic latent space and reconstruction;
- `AutoencAdvE/Ed`: matches latent codes to a prior through adversarial training.
- `SkclaGb`: gradient-boosted decision-tree classification;
- `TorchClaMlp`: configurable PyTorch MLP classification;
- `TorchRegMlp`: configurable PyTorch MLP regression.
- `TorchTsMlp`: sliding-window time-series forecasting with a dense PyTorch MLP;
- `TsConv1D`: sliding-window forecasting with one-dimensional convolutions;
- `TsLstm`: sliding-window forecasting with recurrent LSTM layers.

## Installation

Install the published package from PyPI:

```bash
python -m pip install pydaltoolboxdp
```

Development installation from this checkout:

```bash
python -m pip install -e .
```

On Python 3.14, prefer binary dependencies so `pyharbinger` selects the available
pure-Python `ruptures` wheel instead of attempting a local C compilation:

```bash
python -m pip install --prefer-binary pydaltoolboxdp
```

PyTorch is a core dependency because Deep ++ is specifically the deep-learning
extension layer. GPU use is optional; models run on CPU by default.

## Compatibility

| `pydaltoolboxdp` | `daltoolboxdp` | Implemented families |
|---|---|---|
| 0.1.x | 1.3.767 or newer | dense, denoising, and Conv1D autoencoders |
| 0.2.x | development checkout | adds LSTM autoencoders |
| 0.3.x | development checkout | completes all 14 autoencoder classes |
| 0.4.x | development checkout | adds tabular classification and regression |
| 0.5.x | 1.3.767 from GitHub `main` | adds one-step and multi-step time-series forecasting |
