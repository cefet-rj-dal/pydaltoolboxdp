# LSTM autoencoder — encode

`AutoencLstmE` interprets every observation window as a temporal sequence and learns
a fixed-size latent representation with an LSTM encoder.


```python
import pandas as pd
import pytspredit as ts

from pydaltoolboxdp import AutoencLstmE

windows = ts.ts_data(ts.datasets.tsd()["y"], sw=5)
preprocess = ts.TsNormGminmax().fit(windows)
windows = preprocess.transform(windows)
sample = ts.ts_sample(windows, test_size=10)
train, test = sample.train, sample.test
```

## Learn temporal representations


```python
model = AutoencLstmE(
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
      <td>0.486898</td>
      <td>0.105058</td>
      <td>0.219986</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.491615</td>
      <td>0.104461</td>
      <td>0.218938</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.491934</td>
      <td>0.105200</td>
      <td>0.218387</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.487840</td>
      <td>0.107253</td>
      <td>0.218383</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.479482</td>
      <td>0.110493</td>
      <td>0.219009</td>
    </tr>
  </tbody>
</table>
</div>




```python
encoded.shape
```




    (10, 3)



With `sequence_length=5`, each five-column window becomes five time steps with one
feature per step. Other factorizations are accepted when `sequence_length` divides
`input_size`.

## References

- Hochreiter, S., and Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural
  Computation, 9(8), 1735–1780.
