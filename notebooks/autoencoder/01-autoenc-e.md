---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Dense autoencoder — encode

`AutoencE` learns a nonlinear compression of time-series windows. This mirrors the
`daltoolboxdp::autoenc_e()` example: build windows, normalize them, split train/test, fit
the autoencoder, and inspect the latent representation.

## Data and preprocessing

```python
import pandas as pd
import pytspredit as ts

from pydaltoolboxdp import AutoencE

windows = ts.ts_data(ts.datasets.tsd()["y"], sw=5)
preprocess = ts.TsNormGminmax().fit(windows)
windows = preprocess.transform(windows)
sample = ts.ts_sample(windows, test_size=10)
train, test = sample.train, sample.test

train.head()
```

## Fit a 5 → 3 encoder

The explicit seed makes the example reproducible on CPU.

```python
model = AutoencE(
    input_size=5,
    encoding_size=3,
    epochs=15,
    batch_size=8,
    random_state=7,
).fit(train)

pd.DataFrame({"train_loss": model.train_loss_}).tail()
```

## Transform unseen windows

```python
encoded = model.transform(test)
pd.DataFrame(encoded, columns=["z1", "z2", "z3"]).head()
```

```python
encoded.shape
```

The five lag dimensions are represented by three learned latent attributes. Consumers
can use `AutoencE` anywhere an `AutoencBaseE` transformation is accepted.

## References

- Hinton, G. E., and Salakhutdinov, R. R. (2006). *Reducing the Dimensionality of Data
  with Neural Networks.* Science, 313(5786), 504–507.
- Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep
  Learning Library.* NeurIPS 32.
