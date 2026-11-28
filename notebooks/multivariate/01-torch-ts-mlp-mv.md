# Multivariate forecasting with PyTorch MLP

```python
import numpy as np
import pandas as pd
import pytspredit as ts
from pydaltoolboxdp import TorchTsMlp

y = ts.datasets.tsd()["y"].to_numpy()
x1 = np.concatenate([y[1:], y[-1:]])
mv = ts.ts_data_mv(pd.DataFrame({"y": y, "x1": x1}), y="y")
sample = ts.ts_sample(mv, test_size=5)
make = lambda seed: TorchTsMlp(hidden_sizes=[6], dropout=0.0, epochs=5,
                                batch_size=8, learning_rate=0.01, random_state=seed)
model = ts.TsRegswMv(
    model_y=ts.ts_mv_spec(make(43), variables=["y", "x1"]),
    models_x={"x1": ts.ts_mv_spec(make(47))}, window_size=5,
).fit(sample.train)
prediction = model.predict(steps_ahead=5)
prediction.system
```

```python
ts.plot_ts_pred_mv(sample.train, sample.test, prediction, variable="y")
```
