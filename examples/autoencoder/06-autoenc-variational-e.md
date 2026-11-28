# Variational autoencoder — encode

`AutoencVariationalE` learns distribution parameters for a regularized latent space.


```python
import pandas as pd
import pytspredit as ts

from pydaltoolboxdp import AutoencVariationalE

windows = ts.ts_data(ts.datasets.tsd()["y"], sw=5)
preprocess = ts.TsNormGminmax().fit(windows)
windows = preprocess.transform(windows)
sample = ts.ts_sample(windows, test_size=10)
train, test = sample.train, sample.test
```


```python
model = AutoencVariationalE(
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
encoded = model.transform(test)
pd.DataFrame(encoded, columns=["mean1", "mean2", "mean3", "var1", "var2", "var3"]).head()
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
      <th>mean1</th>
      <th>mean2</th>
      <th>mean3</th>
      <th>var1</th>
      <th>var2</th>
      <th>var3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.426355</td>
      <td>-0.077034</td>
      <td>-0.151110</td>
      <td>0.042599</td>
      <td>0.064013</td>
      <td>-0.472480</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.426242</td>
      <td>-0.070302</td>
      <td>-0.147690</td>
      <td>0.036782</td>
      <td>0.059165</td>
      <td>-0.484906</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.426676</td>
      <td>-0.065449</td>
      <td>-0.146075</td>
      <td>0.033012</td>
      <td>0.055093</td>
      <td>-0.493880</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.427631</td>
      <td>-0.062776</td>
      <td>-0.146366</td>
      <td>0.031525</td>
      <td>0.052053</td>
      <td>-0.498844</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.429046</td>
      <td>-0.062449</td>
      <td>-0.148545</td>
      <td>0.032413</td>
      <td>0.050231</td>
      <td>-0.499490</td>
    </tr>
  </tbody>
</table>
</div>



The six output columns contain the mean and scale parameters for three latent dimensions.

## References

- Kingma, D. P., and Welling, M. (2014). *Auto-Encoding Variational Bayes.* ICLR.
