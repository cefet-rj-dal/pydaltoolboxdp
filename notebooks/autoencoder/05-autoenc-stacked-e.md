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

# Stacked autoencoder — encode

`AutoencStackedE` trains a sequence of dense autoencoders greedily. Each stage receives
the latent output of the previous stage and learns a progressively smaller representation.

```python
import pandas as pd
import pytspredit as ts

from pydaltoolboxdp import AutoencStackedE

windows = ts.ts_data(ts.datasets.tsd()["y"], sw=5)
preprocess = ts.TsNormGminmax().fit(windows)
windows = preprocess.transform(windows)
sample = ts.ts_sample(windows, test_size=10)
train, test = sample.train, sample.test
```

## Learn hierarchical representations

```python
model = AutoencStackedE(
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
encoded = model.transform(test)
pd.DataFrame(encoded, columns=["z1", "z2", "z3"]).head()
```

```python
encoded.shape
```

`encoding_sizes=[4, 3]` defines two stages. The exposed loss is the final stage's
history, matching the original Deep ++ interface.

## References

- Bengio, Y., Lamblin, P., Popovici, D., and Larochelle, H. (2007). *Greedy
  Layer-Wise Training of Deep Networks.* NeurIPS 19.
