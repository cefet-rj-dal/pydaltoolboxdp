# Convolutional autoencoder — encode and decode

`AutoencConvEd` learns local patterns in each window and reconstructs the observations
through transposed one-dimensional convolutions.


```python
import numpy as np
import pandas as pd
import pytspredit as ts

from pydaltoolboxdp import AutoencConvEd

windows = ts.ts_data(ts.datasets.tsd()["y"], sw=5)
preprocess = ts.TsNormGminmax().fit(windows)
windows = preprocess.transform(windows)
sample = ts.ts_sample(windows, test_size=10)
train, test = sample.train, sample.test
```

## Fit and reconstruct


```python
model = AutoencConvEd(
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
reconstructed = model.transform(test)
reconstructed = pd.DataFrame(reconstructed, columns=test.columns, index=test.index)
reconstructed.head()
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
      <th>t4</th>
      <th>t3</th>
      <th>t2</th>
      <th>t1</th>
      <th>t0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.564895</td>
      <td>0.619541</td>
      <td>0.556990</td>
      <td>0.570808</td>
      <td>0.585790</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.565383</td>
      <td>0.625305</td>
      <td>0.560331</td>
      <td>0.571880</td>
      <td>0.588586</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.566725</td>
      <td>0.628096</td>
      <td>0.560677</td>
      <td>0.571880</td>
      <td>0.590351</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.567159</td>
      <td>0.628495</td>
      <td>0.560612</td>
      <td>0.571897</td>
      <td>0.590924</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.566518</td>
      <td>0.626551</td>
      <td>0.560506</td>
      <td>0.572117</td>
      <td>0.590302</td>
    </tr>
  </tbody>
</table>
</div>




```python
mse = ((test - reconstructed) ** 2).mean()
r2 = {
    column: np.corrcoef(test[column], reconstructed[column])[0, 1] ** 2
    for column in test.columns
}
pd.DataFrame({"mse": mse, "R2": pd.Series(r2)})
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
      <th>mse</th>
      <th>R2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>t4</th>
      <td>0.111119</td>
      <td>0.468043</td>
    </tr>
    <tr>
      <th>t3</th>
      <td>0.078645</td>
      <td>0.912472</td>
    </tr>
    <tr>
      <th>t2</th>
      <td>0.105143</td>
      <td>0.967898</td>
    </tr>
    <tr>
      <th>t1</th>
      <td>0.090441</td>
      <td>0.776503</td>
    </tr>
    <tr>
      <th>t0</th>
      <td>0.080302</td>
      <td>0.720854</td>
    </tr>
  </tbody>
</table>
</div>



The reconstruction has the same two-dimensional shape as the original windows and is
bounded by the decoder's sigmoid output.

## References

- Masci, J., Meier, U., Ciresan, D., and Schmidhuber, J. (2011). *Stacked
  Convolutional Auto-Encoders for Hierarchical Feature Extraction.* ICANN, 52–59.
