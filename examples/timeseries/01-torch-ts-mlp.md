# PyTorch MLP time-series forecasting

`TorchTsMlp` specializes `pytspredit.TsRegSW`, so preprocessing, lag mapping and
recursive multi-step forecasting are inherited.


```python
import numpy as np
import pytspredit as ts
from pydaltoolboxdp import TorchTsMlp

windows = ts.ts_data(ts.datasets.tsd()["y"], 6)
sample = ts.ts_sample(windows, test_size=5)
train, test = ts.ts_projection(sample.train), ts.ts_projection(sample.test)
model = TorchTsMlp(
    ts.TsNormGminmax(), input_size=5, hidden_sizes=[8, 4], dropout=0.0,
    epochs=8, batch_size=8, learning_rate=0.01, random_state=31,
).fit(train.input, train.output)
prediction = model.predict(test.input.iloc[[0]], steps_ahead=5)
model.evaluate(test.output.to_numpy().ravel(), prediction)["metrics"]
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
      <th>smape</th>
      <th>R2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.038344</td>
      <td>1.622778</td>
      <td>-7.968232</td>
    </tr>
  </tbody>
</table>
</div>




```python
adjusted = model.predict(train.input)
y = np.concatenate([train.output.to_numpy().ravel(), test.output.to_numpy().ravel()])
ts.plot_ts_pred(y=y, yadj=adjusted, ypred=prediction)
```




    
![png](fig/01-torch-ts-mlp/output_2_0.png)
    




    
![png](fig/01-torch-ts-mlp/output_2_1.png)
    


References: Zhang, Patuwo & Hu (1998); Paszke et al. (2019).
