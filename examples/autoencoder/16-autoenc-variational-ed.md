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
      <td>24.984897</td>
    </tr>
    <tr>
      <th>11</th>
      <td>24.902713</td>
    </tr>
    <tr>
      <th>12</th>
      <td>24.833298</td>
    </tr>
    <tr>
      <th>13</th>
      <td>24.760927</td>
    </tr>
    <tr>
      <th>14</th>
      <td>24.693063</td>
    </tr>
  </tbody>
</table>
</div>




```python
reconstructed = pd.DataFrame(model.transform(test), columns=test.columns, index=test.index)
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
      <td>0.533502</td>
      <td>0.536917</td>
      <td>0.434209</td>
      <td>0.452916</td>
      <td>0.460876</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.533818</td>
      <td>0.536303</td>
      <td>0.433620</td>
      <td>0.453655</td>
      <td>0.460869</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.533778</td>
      <td>0.536397</td>
      <td>0.433709</td>
      <td>0.453542</td>
      <td>0.460885</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.533574</td>
      <td>0.536811</td>
      <td>0.434105</td>
      <td>0.453045</td>
      <td>0.460906</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.533428</td>
      <td>0.537009</td>
      <td>0.434250</td>
      <td>0.452863</td>
      <td>0.460898</td>
    </tr>
  </tbody>
</table>
</div>




```python
mse = ((test - reconstructed) ** 2).mean()
r2 = {column: np.corrcoef(test[column], reconstructed[column])[0, 1] ** 2 for column in test.columns}
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
      <td>0.131548</td>
      <td>0.112442</td>
    </tr>
    <tr>
      <th>t3</th>
      <td>0.126073</td>
      <td>0.070526</td>
    </tr>
    <tr>
      <th>t2</th>
      <td>0.189070</td>
      <td>0.053577</td>
    </tr>
    <tr>
      <th>t1</th>
      <td>0.154336</td>
      <td>0.052552</td>
    </tr>
    <tr>
      <th>t0</th>
      <td>0.128941</td>
      <td>0.478314</td>
    </tr>
  </tbody>
</table>
</div>



## References

- Kingma, D. P., and Welling, M. (2014). *Auto-Encoding Variational Bayes.* ICLR.
