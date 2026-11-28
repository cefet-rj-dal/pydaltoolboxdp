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

# Variational autoencoder — encode and decode

```python
import numpy as np
import pandas as pd
import pytspredit as ts

from pydaltoolboxdp import AutoencVariationalEd

windows = ts.ts_data(ts.datasets.tsd()["y"], sw=5)
preprocess = ts.TsNormGminmax().fit(windows)
windows = preprocess.transform(windows)
sample = ts.ts_sample(windows, test_size=10)
train, test = sample.train, sample.test
```

```python
model = AutoencVariationalEd(
    input_size=5, encoding_size=3,
    encoder_hidden_sizes=[8, 4], decoder_hidden_sizes=[4, 8],
    output_activation="sigmoid", reconstruction_loss="bce",
    epochs=15, batch_size=8, random_state=42,
).fit(train)
pd.DataFrame({"train_loss": model.train_loss_}).tail()
```

```python
reconstructed = pd.DataFrame(model.transform(test), columns=test.columns, index=test.index)
reconstructed.head()
```

```python
mse = ((test - reconstructed) ** 2).mean()
r2 = {column: np.corrcoef(test[column], reconstructed[column])[0, 1] ** 2 for column in test.columns}
pd.DataFrame({"mse": mse, "R2": pd.Series(r2)})
```

## References

- Kingma, D. P., and Welling, M. (2014). *Auto-Encoding Variational Bayes.* ICLR.
