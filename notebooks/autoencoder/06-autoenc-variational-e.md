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

# Variational autoencoder — encode

`AutoencVariationalE` learns distribution parameters for a regularized latent space.

```python
import pandas as pd
import pytspredit as ts

from pydaltoolboxdp import AutoencVariationalE

windows = ts.ts_data(ts.datasets.tsd()["y"], sw=5)
preprocess = ts.TsNormGminmax().fit(windows)
windows = preprocess.transform(windows)
sample = ts.ts_sample(windows, test_size=10)
train, test = sample.train, sample.test
```

```python
model = AutoencVariationalE(
    input_size=5, encoding_size=3,
    encoder_hidden_sizes=[8, 4], decoder_hidden_sizes=[4, 8],
    output_activation="sigmoid", reconstruction_loss="bce",
    epochs=15, batch_size=8, random_state=42,
).fit(train)
pd.DataFrame({"train_loss": model.train_loss_}).tail()
```

```python
encoded = model.transform(test)
pd.DataFrame(encoded, columns=["mean1", "mean2", "mean3", "var1", "var2", "var3"]).head()
```

The six output columns contain the mean and scale parameters for three latent dimensions.

## References

- Kingma, D. P., and Welling, M. (2014). *Auto-Encoding Variational Bayes.* ICLR.
