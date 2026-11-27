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

# Denoising autoencoder — encode and decode

`AutoencDenoiseEd` learns to reconstruct clean observations from corrupted training
inputs. Its output preserves the original window shape.

```python
import numpy as np
import pandas as pd
import pytspredit as ts

from pydaltoolboxdp import AutoencDenoiseEd

windows = ts.ts_data(ts.datasets.tsd()["y"], sw=5)
preprocess = ts.TsNormGminmax().fit(windows)
windows = preprocess.transform(windows)
sample = ts.ts_sample(windows, test_size=10)
train, test = sample.train, sample.test
```

## Fit and reconstruct

```python
model = AutoencDenoiseEd(
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
reconstructed = model.transform(test)
reconstructed = pd.DataFrame(reconstructed, columns=test.columns, index=test.index)
reconstructed.head()
```

```python
mse = ((test - reconstructed) ** 2).mean()
r2 = {
    column: np.corrcoef(test[column], reconstructed[column])[0, 1] ** 2
    for column in test.columns
}
pd.DataFrame({"mse": mse, "R2": pd.Series(r2)})
```

The reconstruction is evaluated against the clean test windows, matching the scientific
objective used during training.

## References

- Vincent, P., Larochelle, H., Bengio, Y., and Manzagol, P. A. (2008). *Extracting and
  Composing Robust Features with Denoising Autoencoders.* ICML, 1096–1103.
