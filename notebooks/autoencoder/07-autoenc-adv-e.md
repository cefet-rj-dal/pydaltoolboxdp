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

# Adversarial autoencoder — encode

`AutoencAdvE` regularizes encoded observations against a Gaussian prior.

```python
import pandas as pd
import pytspredit as ts

from pydaltoolboxdp import AutoencAdvE

windows = ts.ts_data(ts.datasets.tsd()["y"], sw=5)
preprocess = ts.TsNormGminmax().fit(windows)
windows = preprocess.transform(windows)
sample = ts.ts_sample(windows, test_size=10)
train, test = sample.train, sample.test
```

```python
model = AutoencAdvE(
    input_size=5, encoding_size=3,
    encoder_hidden_sizes=[8], decoder_hidden_sizes=[8],
    discriminator_hidden_sizes=[8], dropout=0.1, latent_prior_scale=1.5,
    epochs=15, batch_size=8, random_state=42,
).fit(train)
pd.DataFrame({"train_loss": model.train_loss_}).tail()
```

```python
pd.DataFrame(model.transform(test), columns=["z1", "z2", "z3"]).head()
```

## References

- Makhzani, A., et al. (2016). *Adversarial Autoencoders.* ICLR Workshop.
