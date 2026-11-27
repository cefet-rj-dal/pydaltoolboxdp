# Denoising autoencoder — encode

`AutoencDenoiseE` corrupts training inputs with Gaussian noise while preserving clean
targets. The resulting latent representation is useful when measurements are noisy.


```python
import pandas as pd
import pytspredit as ts

from pydaltoolboxdp import AutoencDenoiseE

windows = ts.ts_data(ts.datasets.tsd()["y"], sw=5)
preprocess = ts.TsNormGminmax().fit(windows)
windows = preprocess.transform(windows)
sample = ts.ts_sample(windows, test_size=10)
train, test = sample.train, sample.test
```

## Learn robust latent attributes


```python
model = AutoencDenoiseE(
    input_size=5,
    encoding_size=3,
    noise_factor=0.2,
    epochs=15,
    batch_size=8,
    random_state=7,
).fit(train)

pd.DataFrame({"train_loss": model.train_loss_}).tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>train_loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>0.105958</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.099589</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.086625</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.069737</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.061609</td>
    </tr>
  </tbody>
</table>
</div>




```python
encoded = model.transform(test)
pd.DataFrame(encoded, columns=["z1", "z2", "z3"]).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>z1</th>
      <th>z2</th>
      <th>z3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.843300</td>
      <td>-0.277194</td>
      <td>0.707830</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.885109</td>
      <td>-0.286587</td>
      <td>0.740641</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.900628</td>
      <td>-0.285406</td>
      <td>0.756966</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.888659</td>
      <td>-0.273865</td>
      <td>0.755660</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.847749</td>
      <td>-0.254023</td>
      <td>0.735582</td>
    </tr>
  </tbody>
</table>
</div>




```python
encoded.shape
```




    (10, 3)



The corruption is used only during fitting. Transforming unseen windows is deterministic
and returns their clean latent representation.

## References

- Vincent, P., Larochelle, H., Bengio, Y., and Manzagol, P. A. (2008). *Extracting and
  Composing Robust Features with Denoising Autoencoders.* ICML, 1096–1103.
