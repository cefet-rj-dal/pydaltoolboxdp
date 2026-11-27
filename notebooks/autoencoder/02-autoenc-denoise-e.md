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

# Denoising autoencoder — encode

`AutoencDenoiseE` corrupts training inputs with Gaussian noise while preserving clean
targets. The resulting latent representation is useful when measurements are noisy.

```python
import pandas as pd
import pytspredit as ts

from pydaltoolboxdp import AutoencDenoiseE

windows = ts.ts_data(ts.datasets.tsd()["y"], sw=5)
preprocess = ts.TsNormGminmax().fit(windows)
windows = preprocess.transform(windows)
sample = ts.ts_sample(windows, test_size=10)
train, test = sample.train, sample.test
```

## Learn robust latent attributes

```python
model = AutoencDenoiseE(
    input_size=5,
    encoding_size=3,
    noise_factor=0.2,
    epochs=15,
    batch_size=8,
    random_state=7,
).fit(train)

pd.DataFrame({"train_loss": model.train_loss_}).tail()
```

```python
encoded = model.transform(test)
pd.DataFrame(encoded, columns=["z1", "z2", "z3"]).head()
```

```python
encoded.shape
```

The corruption is used only during fitting. Transforming unseen windows is deterministic
and returns their clean latent representation.

## References

- Vincent, P., Larochelle, H., Bengio, Y., and Manzagol, P. A. (2008). *Extracting and
  Composing Robust Features with Denoising Autoencoders.* ICML, 1096–1103.
