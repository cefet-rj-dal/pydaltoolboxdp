# PyTorch MLP regressor

`TorchRegMlp` infers the predictor count at fit time and returns one numeric prediction
per observation. This mirrors `daltoolboxdp::torch_reg_mlp()`.

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pydaltoolboxdp import TorchRegMlp

data = load_diabetes(as_frame=True).frame.rename(columns={"target": "disease_progression"})
train, test = train_test_split(data, test_size=0.25, random_state=7)
features = [column for column in data.columns if column != "disease_progression"]
scaler = StandardScaler().fit(train[features])
train.loc[:, features] = scaler.transform(train[features])
test.loc[:, features] = scaler.transform(test[features])

model = TorchRegMlp(
    "disease_progression", hidden_sizes=[16, 8], dropout=0.0,
    normalization="layer", epochs=40, batch_size=32, learning_rate=0.01,
    random_state=17,
).fit(train)
prediction = model.predict(test)
evaluation = model.evaluate(test["disease_progression"], prediction)
evaluation["metrics"]
```

```python
model.input_size, model.epochs_done_, model.train_loss_[-1]
```

## References

- Goodfellow, I., Bengio, Y., and Courville, A. (2016). *Deep Learning.* MIT Press.
- Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep
  Learning Library.* NeurIPS 32.
