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

# LSTM autoencoder — encode and decode

`AutoencLstmEd` compresses a temporal window with an LSTM encoder and reconstructs
the complete sequence with an LSTM decoder.

```python
import numpy as np
import pandas as pd
import pytspredit as ts

from pydaltoolboxdp import AutoencLstmEd

windows = ts.ts_data(ts.datasets.tsd()["y"], sw=5)
preprocess = ts.TsNormGminmax().fit(windows)
windows = preprocess.transform(windows)
sample = ts.ts_sample(windows, test_size=10)
train, test = sample.train, sample.test
```

## Fit and reconstruct

```python
model = AutoencLstmEd(
    input_size=5,
    encoding_size=3,
    sequence_length=5,
    lstm_hidden_size=8,
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

The reconstruction returns the original two-dimensional window shape; MSE and R²
measure how much temporal information the latent state preserved.

## References

- Hochreiter, S., and Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural
  Computation, 9(8), 1735–1780.
