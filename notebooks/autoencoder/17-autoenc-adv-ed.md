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

# Adversarial autoencoder — encode and decode

```python
import numpy as np
import pandas as pd
import pytspredit as ts

from pydaltoolboxdp import AutoencAdvEd

windows = ts.ts_data(ts.datasets.tsd()["y"], sw=5)
preprocess = ts.TsNormGminmax().fit(windows)
windows = preprocess.transform(windows)
sample = ts.ts_sample(windows, test_size=10)
train, test = sample.train, sample.test
```

```python
model = AutoencAdvEd(
    input_size=5, encoding_size=3,
    encoder_hidden_sizes=[8], decoder_hidden_sizes=[8],
    discriminator_hidden_sizes=[8], dropout=0.1, latent_prior_scale=1.5,
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

- Makhzani, A., et al. (2016). *Adversarial Autoencoders.* ICLR Workshop.
