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

```python
ts.plot_ts_pred_mv(sample.train, sample.test, prediction, variable="y")
```
