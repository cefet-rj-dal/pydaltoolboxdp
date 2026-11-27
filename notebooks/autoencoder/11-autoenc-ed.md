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

# Dense autoencoder — encode and decode

`AutoencEd` exposes the complete encode/decode path. Its transformation has the same
shape as the input, which makes reconstruction error available to denoising and anomaly
detection workflows.

## Data and preprocessing

```python
import numpy as np
import pandas as pd
import pytspredit as ts

from pydaltoolboxdp import AutoencEd

windows = ts.ts_data(ts.datasets.tsd()["y"], sw=5)
preprocess = ts.TsNormGminmax().fit(windows)
windows = preprocess.transform(windows)
sample = ts.ts_sample(windows, test_size=10)
train, test = sample.train, sample.test

train.head()
```

## Fit a 5 → 3 → 5 autoencoder

```python
model = AutoencEd(
    input_size=5,
    encoding_size=3,
    epochs=15,
    batch_size=8,
    random_state=7,
).fit(train)

pd.DataFrame({"train_loss": model.train_loss_}).tail()
```

## Reconstruct unseen windows

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

The per-attribute scores reveal which positions of the temporal window are reconstructed
most faithfully. `HanAutoencoder` and `HcpWaypoint` will reuse this same contract in the
Deep ++ event-detection milestone.

## References

- Hinton, G. E., and Salakhutdinov, R. R. (2006). *Reducing the Dimensionality of Data
  with Neural Networks.* Science, 313(5786), 504–507.
- Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep
  Learning Library.* NeurIPS 32.
