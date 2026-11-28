# PyTorch MLP classifier

`TorchClaMlp` exposes configurable hidden layers, activation, normalization, validation,
and stopping while retaining the `pydaltoolbox` classification workflow.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pydaltoolboxdp import TorchClaMlp

iris = load_iris(as_frame=True).frame.rename(columns={"target": "class"})
iris["class"] = iris["class"].map(dict(enumerate(load_iris().target_names)))
train, test = train_test_split(iris, test_size=0.25, random_state=7, stratify=iris["class"])
features = [column for column in iris.columns if column != "class"]
scaler = StandardScaler().fit(train[features])
train.loc[:, features] = scaler.transform(train[features])
test.loc[:, features] = scaler.transform(test[features])

levels = list(load_iris().target_names)
model = TorchClaMlp(
    "class", levels, input_size=4, hidden_sizes=[16, 8], dropout=0.0,
    normalization="layer", epochs=40, batch_size=16, learning_rate=0.01,
    random_state=13,
).fit(train)
prediction = model.predict(test)
evaluation = model.evaluate(test["class"], prediction)
evaluation["metrics"]
```

```python
model.epochs_done_, model.train_loss_[-1]
```

## References

- Goodfellow, I., Bengio, Y., and Courville, A. (2016). *Deep Learning.* MIT Press.
- Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep
  Learning Library.* NeurIPS 32.
