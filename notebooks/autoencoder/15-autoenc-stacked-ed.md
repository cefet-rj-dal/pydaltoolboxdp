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

# Stacked autoencoder — encode and decode

`AutoencStackedEd` encodes through every greedily trained stage and decodes through
the same stages in reverse order to reconstruct the original window.

```python
import numpy as np
import pandas as pd
import pytspredit as ts

from pydaltoolboxdp import AutoencStackedEd

windows = ts.ts_data(ts.datasets.tsd()["y"], sw=5)
preprocess = ts.TsNormGminmax().fit(windows)
windows = preprocess.transform(windows)
sample = ts.ts_sample(windows, test_size=10)
train, test = sample.train, sample.test
```

## Fit and reconstruct

```python
model = AutoencStackedEd(
    input_size=5,
    encoding_size=3,
    encoding_sizes=[4, 3],
    encoder_hidden_sizes=[8],
    decoder_hidden_sizes=[8],
    epochs=15,
    batch_size=8,
    random_state=42,
).fit(train)

pd.DataFrame({"last_stage_loss": model.train_loss_}).tail()
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

MSE and R² evaluate how much information remains after both hierarchical bottlenecks.

## References

- Bengio, Y., Lamblin, P., Popovici, D., and Larochelle, H. (2007). *Greedy
  Layer-Wise Training of Deep Networks.* NeurIPS 19.
