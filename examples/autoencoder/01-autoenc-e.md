# Dense autoencoder — encode

`AutoencE` learns a nonlinear compression of time-series windows. This mirrors the
`daltoolboxdp::autoenc_e()` example: build windows, normalize them, split train/test, fit
the autoencoder, and inspect the latent representation.

## Data and preprocessing


```python
import pandas as pd
import pytspredit as ts

from pydaltoolboxdp import AutoencE

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



## Fit a 5 → 3 encoder

The explicit seed makes the example reproducible on CPU.


```python
model = AutoencE(
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



## Transform unseen windows


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
      <td>-0.841607</td>
      <td>-0.354669</td>
      <td>0.697667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.885849</td>
      <td>-0.366033</td>
      <td>0.727122</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.902482</td>
      <td>-0.364969</td>
      <td>0.741017</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.889191</td>
      <td>-0.352207</td>
      <td>0.738428</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.843949</td>
      <td>-0.329595</td>
      <td>0.716637</td>
    </tr>
  </tbody>
</table>
</div>




```python
encoded.shape
```




    (10, 3)



The five lag dimensions are represented by three learned latent attributes. Consumers
can use `AutoencE` anywhere an `AutoencBaseE` transformation is accepted.

## References

- Hinton, G. E., and Salakhutdinov, R. R. (2006). *Reducing the Dimensionality of Data
  with Neural Networks.* Science, 313(5786), 504–507.
- Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep
  Learning Library.* NeurIPS 32.
