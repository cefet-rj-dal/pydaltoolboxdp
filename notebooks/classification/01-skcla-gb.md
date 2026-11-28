# Gradient boosting classifier

`SkclaGb` fits an additive ensemble of decision trees and returns class probabilities in
the requested level order. This is the Python counterpart of
`daltoolboxdp::skcla_gb()`.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from pydaltoolboxdp import SkclaGb

iris = load_iris(as_frame=True).frame.rename(columns={"target": "class"})
iris["class"] = iris["class"].map(dict(enumerate(load_iris().target_names)))
train, test = train_test_split(iris, test_size=0.25, random_state=7, stratify=iris["class"])

levels = list(load_iris().target_names)
model = SkclaGb("class", levels, n_estimators=80, learning_rate=0.05,
                max_depth=2, random_state=11).fit(train)
prediction = model.predict(test)
evaluation = model.evaluate(test["class"], prediction)
evaluation["metrics"]
```

The prediction is a probability table whose columns are exactly `levels`.

## References

- Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting
  Machine.* Annals of Statistics, 29(5), 1189–1232.
