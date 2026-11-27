# Dense autoencoder — encode and decode

`AutoencEd` exposes the complete encode/decode path. Its transformation has the same
shape as the input, which makes reconstruction error available to denoising and anomaly
detection workflows.

## Data and preprocessing


```python
import numpy as np
import pandas as pd
import pytspredit as ts

from pydaltoolboxdp import AutoencEd

windows = ts.ts_data(ts.datasets.tsd()["y"], sw=5)
preprocess = ts.TsNormGminmax().fit(windows)
windows = preprocess.transform(windows)
sample = ts.ts_sample(windows, test_size=10)
train, test = sample.train, sample.test

train.head()
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
      <td>0.500450</td>
      <td>0.624351</td>
      <td>0.740549</td>
      <td>0.841818</td>
      <td>0.921863</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.624351</td>
      <td>0.740549</td>
      <td>0.841818</td>
      <td>0.921863</td>
      <td>0.975706</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.740549</td>
      <td>0.841818</td>
      <td>0.921863</td>
      <td>0.975706</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.841818</td>
      <td>0.921863</td>
      <td>0.975706</td>
      <td>1.000000</td>
      <td>0.993235</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.921863</td>
      <td>0.975706</td>
      <td>1.000000</td>
      <td>0.993235</td>
      <td>0.955830</td>
    </tr>
  </tbody>
</table>
</div>



## Fit a 5 → 3 → 5 autoencoder


```python
model = AutoencEd(
    input_size=5,
    encoding_size=3,
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
      <td>0.103576</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.087778</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.081352</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.067140</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.065849</td>
    </tr>
  </tbody>
</table>
</div>



## Reconstruct unseen windows


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
      <td>0.627973</td>
      <td>0.814237</td>
      <td>0.716725</td>
      <td>0.919694</td>
      <td>0.719536</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.643378</td>
      <td>0.837144</td>
      <td>0.726848</td>
      <td>0.951452</td>
      <td>0.731207</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.649988</td>
      <td>0.844472</td>
      <td>0.730602</td>
      <td>0.962964</td>
      <td>0.735856</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.645912</td>
      <td>0.834718</td>
      <td>0.727814</td>
      <td>0.953428</td>
      <td>0.731682</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.631786</td>
      <td>0.808821</td>
      <td>0.717498</td>
      <td>0.921024</td>
      <td>0.719628</td>
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
      <td>0.092208</td>
      <td>0.235556</td>
    </tr>
    <tr>
      <th>t3</th>
      <td>0.021087</td>
      <td>0.806183</td>
    </tr>
    <tr>
      <th>t2</th>
      <td>0.038549</td>
      <td>0.983830</td>
    </tr>
    <tr>
      <th>t1</th>
      <td>0.014367</td>
      <td>0.991364</td>
    </tr>
    <tr>
      <th>t0</th>
      <td>0.049736</td>
      <td>0.951112</td>
    </tr>
  </tbody>
</table>
</div>



The per-attribute scores reveal which positions of the temporal window are reconstructed
most faithfully. `HanAutoencoder` and `HcpWaypoint` will reuse this same contract in the
Deep ++ event-detection milestone.

## References

- Hinton, G. E., and Salakhutdinov, R. R. (2006). *Reducing the Dimensionality of Data
  with Neural Networks.* Science, 313(5786), 504–507.
- Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep
  Learning Library.* NeurIPS 32.
