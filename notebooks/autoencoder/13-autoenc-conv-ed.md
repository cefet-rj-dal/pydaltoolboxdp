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

# Convolutional autoencoder — encode and decode

`AutoencConvEd` learns local patterns in each window and reconstructs the observations
through transposed one-dimensional convolutions.

```python
import numpy as np
import pandas as pd
import pytspredit as ts

from pydaltoolboxdp import AutoencConvEd

windows = ts.ts_data(ts.datasets.tsd()["y"], sw=5)
preprocess = ts.TsNormGminmax().fit(windows)
windows = preprocess.transform(windows)
sample = ts.ts_sample(windows, test_size=10)
train, test = sample.train, sample.test
```

## Fit and reconstruct

```python
model = AutoencConvEd(
    input_size=5,
    encoding_size=3,
    hidden_channels=8,
    bottleneck_channels=12,
    kernel_size=3,
    epochs=15,
    batch_size=8,
    random_state=42,
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

The reconstruction has the same two-dimensional shape as the original windows and is
bounded by the decoder's sigmoid output.

## References

- Masci, J., Meier, U., Ciresan, D., and Schmidhuber, J. (2011). *Stacked
  Convolutional Auto-Encoders for Hierarchical Feature Extraction.* ICANN, 52–59.
