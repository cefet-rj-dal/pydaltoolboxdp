# Stacked autoencoder — encode and decode

`AutoencStackedEd` encodes through every greedily trained stage and decodes through
the same stages in reverse order to reconstruct the original window.


```python
import numpy as np
import pandas as pd
import pytspredit as ts

from pydaltoolboxdp import AutoencStackedEd

windows = ts.ts_data(ts.datasets.tsd()["y"], sw=5)
preprocess = ts.TsNormGminmax().fit(windows)
windows = preprocess.transform(windows)
sample = ts.ts_sample(windows, test_size=10)
train, test = sample.train, sample.test
```

## Fit and reconstruct


```python
model = AutoencStackedEd(
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
      <th>last_stage_loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>0.365484</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.388564</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.360413</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.331506</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.317128</td>
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
      <td>0.197526</td>
      <td>-0.017426</td>
      <td>-0.055261</td>
      <td>0.372937</td>
      <td>0.178497</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.197488</td>
      <td>-0.017387</td>
      <td>-0.055380</td>
      <td>0.372753</td>
      <td>0.178389</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.197452</td>
      <td>-0.017375</td>
      <td>-0.055484</td>
      <td>0.372594</td>
      <td>0.178315</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.197420</td>
      <td>-0.017392</td>
      <td>-0.055565</td>
      <td>0.372471</td>
      <td>0.178281</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.197395</td>
      <td>-0.017436</td>
      <td>-0.055619</td>
      <td>0.372391</td>
      <td>0.178288</td>
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
      <td>0.478639</td>
      <td>0.240434</td>
    </tr>
    <tr>
      <th>t3</th>
      <td>0.800182</td>
      <td>0.797915</td>
    </tr>
    <tr>
      <th>t2</th>
      <td>0.817398</td>
      <td>0.093320</td>
    </tr>
    <tr>
      <th>t1</th>
      <td>0.212550</td>
      <td>0.058100</td>
    </tr>
    <tr>
      <th>t0</th>
      <td>0.343655</td>
      <td>0.530681</td>
    </tr>
  </tbody>
</table>
</div>



MSE and R² evaluate how much information remains after both hierarchical bottlenecks.

## References

- Bengio, Y., Lamblin, P., Popovici, D., and Larochelle, H. (2007). *Greedy
  Layer-Wise Training of Deep Networks.* NeurIPS 19.
