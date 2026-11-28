# Convolutional autoencoder — encode

`AutoencConvE` treats each observation window as a one-channel sequence. Conv1D layers
learn local patterns before projecting the result into a compact latent representation.


```python
import pandas as pd
import pytspredit as ts

from pydaltoolboxdp import AutoencConvE

windows = ts.ts_data(ts.datasets.tsd()["y"], sw=5)
preprocess = ts.TsNormGminmax().fit(windows)
windows = preprocess.transform(windows)
sample = ts.ts_sample(windows, test_size=10)
train, test = sample.train, sample.test
```

## Learn local features


```python
model = AutoencConvE(
    input_size=5,
    encoding_size=3,
    hidden_channels=8,
    bottleneck_channels=12,
    kernel_size=3,
    epochs=15,
    batch_size=8,
    random_state=42,
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
      <td>0.121087</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.105739</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.126890</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.102213</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.112784</td>
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
      <td>0.257499</td>
      <td>0.226797</td>
      <td>0.139170</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.283921</td>
      <td>0.270202</td>
      <td>0.182966</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.294668</td>
      <td>0.298243</td>
      <td>0.206953</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.294343</td>
      <td>0.307059</td>
      <td>0.213398</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.283176</td>
      <td>0.296512</td>
      <td>0.202592</td>
    </tr>
  </tbody>
</table>
</div>




```python
encoded.shape
```




    (10, 3)



The input is reshaped internally from `(observations, window)` to
`(observations, channel, window)`; callers continue to use ordinary two-dimensional data.

## References

- Masci, J., Meier, U., Ciresan, D., and Schmidhuber, J. (2011). *Stacked
  Convolutional Auto-Encoders for Hierarchical Feature Extraction.* ICANN, 52–59.
