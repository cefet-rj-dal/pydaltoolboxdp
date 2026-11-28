# Multivariate forecasting with Conv1D


```python
import numpy as np
import pandas as pd
import pytspredit as ts
from pydaltoolboxdp import TsConv1D

y = ts.datasets.tsd()["y"].to_numpy()
x1 = np.concatenate([y[1:], y[-1:]])
mv = ts.ts_data_mv(pd.DataFrame({"y": y, "x1": x1}), y="y")
sample = ts.ts_sample(mv, test_size=5)
make = lambda seed: TsConv1D(conv_channels=[3], kernel_sizes=[2],
                              dense_hidden_sizes=[4], epochs=5, batch_size=8,
                              learning_rate=0.01, random_state=seed)
model = ts.TsRegswMv(
    model_y=ts.ts_mv_spec(make(53), variables=["y", "x1"]),
    models_x={"x1": ts.ts_mv_spec(make(59))}, window_size=5,
).fit(sample.train)
prediction = model.predict(steps_ahead=5)
prediction.system
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
      <th>y</th>
      <th>x1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.265848</td>
      <td>0.339354</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.265848</td>
      <td>0.339354</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.265848</td>
      <td>0.339354</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.265848</td>
      <td>0.339354</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.265848</td>
      <td>0.339354</td>
    </tr>
  </tbody>
</table>
</div>




```python
ts.plot_ts_pred_mv(sample.train, sample.test, prediction, variable="y")
```




    
![png](fig/02-ts-conv1d-mv/output_2_0.png)
    




    
![png](fig/02-ts-conv1d-mv/output_2_1.png)
    

