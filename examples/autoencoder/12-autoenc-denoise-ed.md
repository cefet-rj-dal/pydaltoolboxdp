# Denoising autoencoder — encode and decode

`AutoencDenoiseEd` learns to reconstruct clean observations from corrupted training
inputs. Its output preserves the original window shape.


```python
import numpy as np
import pandas as pd
import pytspredit as ts

from pydaltoolboxdp import AutoencDenoiseEd

windows = ts.ts_data(ts.datasets.tsd()["y"], sw=5)
preprocess = ts.TsNormGminmax().fit(windows)
windows = preprocess.transform(windows)
sample = ts.ts_sample(windows, test_size=10)
train, test = sample.train, sample.test
```

## Fit and reconstruct


```python
model = AutoencDenoiseEd(
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
      <td>0.653742</td>
      <td>0.774899</td>
      <td>0.716162</td>
      <td>0.887792</td>
      <td>0.702839</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.669508</td>
      <td>0.795337</td>
      <td>0.725487</td>
      <td>0.918546</td>
      <td>0.713282</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.676781</td>
      <td>0.802319</td>
      <td>0.728761</td>
      <td>0.930539</td>
      <td>0.717745</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.673395</td>
      <td>0.793950</td>
      <td>0.726171</td>
      <td>0.923402</td>
      <td>0.714372</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.660402</td>
      <td>0.771784</td>
      <td>0.716756</td>
      <td>0.896145</td>
      <td>0.704455</td>
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
      <td>0.077215</td>
      <td>0.249276</td>
    </tr>
    <tr>
      <th>t3</th>
      <td>0.029017</td>
      <td>0.809298</td>
    </tr>
    <tr>
      <th>t2</th>
      <td>0.039065</td>
      <td>0.982576</td>
    </tr>
    <tr>
      <th>t1</th>
      <td>0.014937</td>
      <td>0.987252</td>
    </tr>
    <tr>
      <th>t0</th>
      <td>0.053028</td>
      <td>0.947007</td>
    </tr>
  </tbody>
</table>
</div>



The reconstruction is evaluated against the clean test windows, matching the scientific
objective used during training.

## References

- Vincent, P., Larochelle, H., Bengio, Y., and Manzagol, P. A. (2008). *Extracting and
  Composing Robust Features with Denoising Autoencoders.* ICML, 1096–1103.
