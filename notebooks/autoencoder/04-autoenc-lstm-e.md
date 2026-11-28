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

# LSTM autoencoder — encode

`AutoencLstmE` interprets every observation window as a temporal sequence and learns
a fixed-size latent representation with an LSTM encoder.

```python
import pandas as pd
import pytspredit as ts

from pydaltoolboxdp import AutoencLstmE

windows = ts.ts_data(ts.datasets.tsd()["y"], sw=5)
preprocess = ts.TsNormGminmax().fit(windows)
windows = preprocess.transform(windows)
sample = ts.ts_sample(windows, test_size=10)
train, test = sample.train, sample.test
```

## Learn temporal representations

```python
model = AutoencLstmE(
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
encoded = model.transform(test)
pd.DataFrame(encoded, columns=["z1", "z2", "z3"]).head()
```

```python
encoded.shape
```

With `sequence_length=5`, each five-column window becomes five time steps with one
feature per step. Other factorizations are accepted when `sequence_length` divides
`input_size`.

## References

- Hochreiter, S., and Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural
  Computation, 9(8), 1735–1780.
