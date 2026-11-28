# LSTM autoencoder — encode and decode

`AutoencLstmEd` compresses a temporal window with an LSTM encoder and reconstructs
the complete sequence with an LSTM decoder.


```python
import numpy as np
import pandas as pd
import pytspredit as ts

from pydaltoolboxdp import AutoencLstmEd

windows = ts.ts_data(ts.datasets.tsd()["y"], sw=5)
preprocess = ts.TsNormGminmax().fit(windows)
windows = preprocess.transform(windows)
sample = ts.ts_sample(windows, test_size=10)
train, test = sample.train, sample.test
```

## Fit and reconstruct


```python
model = AutoencLstmEd(
    input_size=5,
    encoding_size=3,
    sequence_length=5,
    lstm_hidden_size=8,
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
      <td>0.423223</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.362886</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.424958</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.374500</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.294621</td>
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
      <td>-0.002474</td>
      <td>0.076868</td>
      <td>0.126402</td>
      <td>0.157034</td>
      <td>0.176202</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.002406</td>
      <td>0.076982</td>
      <td>0.126537</td>
      <td>0.157175</td>
      <td>0.176344</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.002422</td>
      <td>0.076955</td>
      <td>0.126504</td>
      <td>0.157140</td>
      <td>0.176308</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.002521</td>
      <td>0.076788</td>
      <td>0.126306</td>
      <td>0.156931</td>
      <td>0.176096</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.002699</td>
      <td>0.076487</td>
      <td>0.125948</td>
      <td>0.156554</td>
      <td>0.175714</td>
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
      <td>0.792992</td>
      <td>0.147333</td>
    </tr>
    <tr>
      <th>t3</th>
      <td>0.643317</td>
      <td>0.733848</td>
    </tr>
    <tr>
      <th>t2</th>
      <td>0.529186</td>
      <td>0.963698</td>
    </tr>
    <tr>
      <th>t1</th>
      <td>0.432916</td>
      <td>0.999569</td>
    </tr>
    <tr>
      <th>t0</th>
      <td>0.346518</td>
      <td>0.981282</td>
    </tr>
  </tbody>
</table>
</div>



The reconstruction returns the original two-dimensional window shape; MSE and R²
measure how much temporal information the latent state preserved.

## References

- Hochreiter, S., and Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural
  Computation, 9(8), 1735–1780.
