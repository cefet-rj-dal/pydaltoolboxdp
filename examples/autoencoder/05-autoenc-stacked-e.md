# Stacked autoencoder — encode

`AutoencStackedE` trains a sequence of dense autoencoders greedily. Each stage receives
the latent output of the previous stage and learns a progressively smaller representation.


```python
import pandas as pd
import pytspredit as ts

from pydaltoolboxdp import AutoencStackedE

windows = ts.ts_data(ts.datasets.tsd()["y"], sw=5)
preprocess = ts.TsNormGminmax().fit(windows)
windows = preprocess.transform(windows)
sample = ts.ts_sample(windows, test_size=10)
train, test = sample.train, sample.test
```

## Learn hierarchical representations


```python
model = AutoencStackedE(
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
      <td>0.609284</td>
      <td>0.382547</td>
      <td>0.454780</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.610761</td>
      <td>0.380953</td>
      <td>0.457942</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.610749</td>
      <td>0.379218</td>
      <td>0.458896</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.609250</td>
      <td>0.377451</td>
      <td>0.457582</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.606357</td>
      <td>0.375761</td>
      <td>0.454083</td>
    </tr>
  </tbody>
</table>
</div>




```python
encoded.shape
```




    (10, 3)



`encoding_sizes=[4, 3]` defines two stages. The exposed loss is the final stage's
history, matching the original Deep ++ interface.

## References

- Bengio, Y., Lamblin, P., Popovici, D., and Larochelle, H. (2007). *Greedy
  Layer-Wise Training of Deep Networks.* NeurIPS 19.
