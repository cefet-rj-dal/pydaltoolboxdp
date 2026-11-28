# Adversarial autoencoder — encode

`AutoencAdvE` regularizes encoded observations against a Gaussian prior.


```python
import pandas as pd
import pytspredit as ts

from pydaltoolboxdp import AutoencAdvE

windows = ts.ts_data(ts.datasets.tsd()["y"], sw=5)
preprocess = ts.TsNormGminmax().fit(windows)
windows = preprocess.transform(windows)
sample = ts.ts_sample(windows, test_size=10)
train, test = sample.train, sample.test
```


```python
model = AutoencAdvE(
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
pd.DataFrame(model.transform(test), columns=["z1", "z2", "z3"]).head()
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
      <td>0.250780</td>
      <td>0.216990</td>
      <td>0.773278</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.260092</td>
      <td>0.246136</td>
      <td>0.850292</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.267418</td>
      <td>0.262907</td>
      <td>0.898845</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.272303</td>
      <td>0.266260</td>
      <td>0.915918</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.274442</td>
      <td>0.255986</td>
      <td>0.900451</td>
    </tr>
  </tbody>
</table>
</div>



## References

- Makhzani, A., et al. (2016). *Adversarial Autoencoders.* ICLR Workshop.
