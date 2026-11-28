# Adversarial autoencoder — encode and decode


```python
import numpy as np
import pandas as pd
import pytspredit as ts

from pydaltoolboxdp import AutoencAdvEd

windows = ts.ts_data(ts.datasets.tsd()["y"], sw=5)
preprocess = ts.TsNormGminmax().fit(windows)
windows = preprocess.transform(windows)
sample = ts.ts_sample(windows, test_size=10)
train, test = sample.train, sample.test
```


```python
model = AutoencAdvEd(
    input_size=5, encoding_size=3,
    encoder_hidden_sizes=[8], decoder_hidden_sizes=[8],
    discriminator_hidden_sizes=[8], dropout=0.1, latent_prior_scale=1.5,
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
      <td>0.128688</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.124102</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.123639</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.120774</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.123351</td>
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
      <td>0.611880</td>
      <td>0.570547</td>
      <td>0.466108</td>
      <td>0.497938</td>
      <td>0.496168</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.615975</td>
      <td>0.570239</td>
      <td>0.468236</td>
      <td>0.495919</td>
      <td>0.498605</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.618633</td>
      <td>0.570035</td>
      <td>0.469604</td>
      <td>0.494614</td>
      <td>0.500198</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.619700</td>
      <td>0.569945</td>
      <td>0.470126</td>
      <td>0.494107</td>
      <td>0.500849</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.619116</td>
      <td>0.569977</td>
      <td>0.469770</td>
      <td>0.494427</td>
      <td>0.500517</td>
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
      <td>0.082206</td>
      <td>0.520385</td>
    </tr>
    <tr>
      <th>t3</th>
      <td>0.104979</td>
      <td>0.669241</td>
    </tr>
    <tr>
      <th>t2</th>
      <td>0.163257</td>
      <td>0.954199</td>
    </tr>
    <tr>
      <th>t1</th>
      <td>0.129336</td>
      <td>0.839844</td>
    </tr>
    <tr>
      <th>t0</th>
      <td>0.111060</td>
      <td>0.722384</td>
    </tr>
  </tbody>
</table>
</div>



## References

- Makhzani, A., et al. (2016). *Adversarial Autoencoders.* ICLR Workshop.
