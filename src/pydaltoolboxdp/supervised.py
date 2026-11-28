"""Deep ++ tabular classifiers and regressors.

The PyTorch learners in this module replace the implementations historically
embedded in ``daltoolboxdp/inst/python`` while implementing the public
``pydaltoolbox`` predictor contracts.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from pydaltoolbox import Classification, Regression, SklearnClassification
from sklearn.ensemble import GradientBoostingClassifier
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset

from ._training import (
    StopController,
    TrainingConfig,
    activation,
    int_list,
    validate_training_options,
)

__all__ = ["SkclaGb", "TorchClaMlp", "TorchRegMlp"]


class SkclaGb(SklearnClassification):
    """Gradient-boosted decision-tree classifier.

    Gradient boosting builds an additive model stage by stage, fitting each new
    tree to the negative gradient of the differentiable loss. ``predict`` returns
    probabilities in ``slevels`` order, as required by the DAL classification
    contract.

    References:
        Friedman, J. H. (2001). Greedy Function Approximation: A Gradient
        Boosting Machine. Annals of Statistics, 29(5):1189-1232.

    Porting:
        Mirrors ``skcla_gb.R`` and replaces ``inst/python/skcla_gb.py``.
        Fidelity: exact scikit-learn estimator and hyperparameter mapping; the
        optional ``random_state`` adds deterministic execution.
    """

    def __init__(
        self,
        attribute: str,
        slevels: Sequence | None = None,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        subsample: float = 1.0,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        loss: str = "log_loss",
        random_state: int | None = None,
    ) -> None:
        super().__init__(attribute, slevels)
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.subsample = float(subsample)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.loss = str(loss)
        self.random_state = random_state

    def _make_model(self) -> GradientBoostingClassifier:
        return GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            loss=self.loss,
            random_state=self.random_state,
        )


def _normalization(kind: str, size: int) -> nn.Module | None:
    kind = str(kind).lower()
    if kind == "none":
        return None
    if kind == "batch":
        return nn.BatchNorm1d(int(size))
    if kind == "layer":
        return nn.LayerNorm(int(size))
    raise ValueError("normalization must be one of {'none', 'batch', 'layer'}")


def _initialize(module: nn.Module, method: str) -> None:
    if not isinstance(module, nn.Linear):
        return
    method = str(method).lower()
    initializers = {
        "xavier_uniform": nn.init.xavier_uniform_,
        "xavier_normal": nn.init.xavier_normal_,
        "kaiming_uniform": lambda value: nn.init.kaiming_uniform_(value, nonlinearity="relu"),
        "kaiming_normal": lambda value: nn.init.kaiming_normal_(value, nonlinearity="relu"),
    }
    if method == "default":
        return
    try:
        initializers[method](module.weight)
    except KeyError as error:
        raise ValueError(f"unsupported init_method: {method}") from error
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class _MlpNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes,
        output_size: int,
        *,
        dropout: float,
        hidden_activation: str,
        output_activation: str,
        normalization: str,
        init_method: str,
        negative_slope: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = int(input_size)
        for hidden_size in int_list(hidden_sizes):
            layers.append(nn.Linear(previous, hidden_size))
            normalizer = _normalization(normalization, hidden_size)
            if normalizer is not None:
                layers.append(normalizer)
            layers.append(activation(hidden_activation, negative_slope))
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            previous = hidden_size
        layers.append(nn.Linear(previous, int(output_size)))
        if output_activation.lower() not in {"none", "identity"}:
            layers.append(activation(output_activation, negative_slope))
        self.net = nn.Sequential(*layers)
        self.net.apply(lambda module: _initialize(module, init_method))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class _TorchMlpMixin:
    def _initialize_mlp(
        self,
        *,
        input_size: int | None,
        hidden_sizes,
        output_size: int,
        dropout: float,
        activation_name: str,
        output_activation: str,
        normalization: str,
        init_method: str,
        epochs: int,
        learning_rate: float,
        lr: float | None,
        validation_strategy: str,
        stopping_rule: str,
        negative_slope: float,
        val_ratio: float,
        batch_size: int,
        patience: int,
        min_delta: float,
        sma_window: int,
        ema_alpha: float,
        test_window: int,
        p_value: float,
        weight_decay: float,
        reshuffle_freq: int,
        random_state: int | None,
        device: str | None,
    ) -> None:
        validation_strategy, stopping_rule = validate_training_options(
            validation_strategy, stopping_rule
        )
        if input_size is not None and int(input_size) <= 0:
            raise ValueError("input_size must be positive")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if not 0 < float(val_ratio) < 1:
            raise ValueError("val_ratio must be between 0 and 1")
        if not 0 <= float(dropout) < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.input_size = None if input_size is None else int(input_size)
        self.hidden_sizes = int_list(hidden_sizes)
        self.output_size = int(output_size)
        self.dropout = float(dropout)
        self.activation = str(activation_name)
        self.output_activation = str(output_activation)
        self.normalization = str(normalization)
        self.init_method = str(init_method)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate if lr is None else lr)
        self.lr = self.learning_rate
        self.validation_strategy = validation_strategy
        self.stopping_rule = stopping_rule
        self.negative_slope = float(negative_slope)
        self.val_ratio = float(val_ratio)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.sma_window = int(sma_window)
        self.ema_alpha = float(ema_alpha)
        self.test_window = int(test_window)
        self.p_value = float(p_value)
        self.weight_decay = float(weight_decay)
        self.reshuffle_freq = max(1, int(reshuffle_freq))
        self.random_state = random_state
        self.device = device or "cpu"
        self.device_ = torch.device(self.device)
        self.model: _MlpNetwork | None = None
        self.network: _MlpNetwork | None = None
        self.train_loss_: list[float] = []
        self.val_loss_: list[float] = []
        self.epochs_done_ = 0

    def _seed(self) -> None:
        if self.random_state is not None:
            np.random.seed(int(self.random_state))
            torch.manual_seed(int(self.random_state))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(self.random_state))

    def _build_model(self) -> None:
        self._seed()
        self.device_ = torch.device(self.device)
        self.model = _MlpNetwork(
            self.input_size,
            self.hidden_sizes,
            self.output_size,
            dropout=self.dropout,
            hidden_activation=self.activation,
            output_activation=self.output_activation,
            normalization=self.normalization,
            init_method=self.init_method,
            negative_slope=self.negative_slope,
        ).float().to(self.device_)
        self.network = self.model

    def _split(self, length: int) -> tuple[np.ndarray, np.ndarray]:
        if length < 2:
            raise ValueError("validation requires at least two observations")
        indices = np.arange(length)
        np.random.shuffle(indices)
        validation_size = min(length - 1, max(1, int(length * self.val_ratio)))
        return indices[validation_size:], indices[:validation_size]

    def _loader(self, features, targets, indices=None, *, shuffle: bool) -> DataLoader:
        if indices is not None:
            features, targets = features[indices], targets[indices]
        return DataLoader(
            TensorDataset(features, targets),
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    def _run_epoch(self, loader: DataLoader, criterion, optimizer=None) -> float:
        losses = []
        self.model.train(optimizer is not None)
        context = torch.enable_grad() if optimizer is not None else torch.no_grad()
        with context:
            for features, targets in loader:
                features = features.to(self.device_)
                targets = targets.to(self.device_)
                if optimizer is not None:
                    optimizer.zero_grad()
                loss = criterion(self.model(features), targets)
                if optimizer is not None:
                    loss.backward()
                    optimizer.step()
                losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else 0.0

    def _train(self, features: torch.Tensor, targets: torch.Tensor, criterion) -> None:
        self._build_model()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        config = TrainingConfig(
            batch_size=self.batch_size,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            validation_strategy=self.validation_strategy,
            stopping_rule=self.stopping_rule,
            val_ratio=self.val_ratio,
            patience=self.patience,
            min_delta=self.min_delta,
            sma_window=self.sma_window,
            ema_alpha=self.ema_alpha,
            test_window=self.test_window,
            p_value=self.p_value,
            reshuffle_freq=self.reshuffle_freq,
        )
        stopper = StopController(config)
        self.train_loss_ = []
        self.val_loss_ = []
        self.epochs_done_ = 0
        train_loader = validation_loader = None
        if self.validation_strategy == "static" and self.stopping_rule != "none":
            train_indices, validation_indices = self._split(len(features))
            train_loader = self._loader(features, targets, train_indices, shuffle=True)
            validation_loader = self._loader(
                features, targets, validation_indices, shuffle=False
            )
        elif self.validation_strategy == "static":
            train_loader = self._loader(features, targets, shuffle=True)

        for epoch in range(self.epochs):
            self.epochs_done_ += 1
            if self.validation_strategy == "dynamic" and (
                epoch == 0 or self.epochs_done_ % self.reshuffle_freq == 0
            ):
                train_indices, validation_indices = self._split(len(features))
                train_loader = self._loader(features, targets, train_indices, shuffle=True)
                validation_loader = self._loader(
                    features, targets, validation_indices, shuffle=False
                )
            self.train_loss_.append(self._run_epoch(train_loader, criterion, optimizer))
            if validation_loader is not None:
                value = self._run_epoch(validation_loader, criterion)
                self.val_loss_.append(value)
                if stopper.step(self.model, value):
                    break
        if stopper.best_state is not None:
            self.model.load_state_dict(stopper.best_state)

    @property
    def train_loss_hist(self) -> list[float]:
        return self.train_loss_

    @property
    def val_loss_hist(self) -> list[float]:
        return self.val_loss_

    @property
    def epochs_done(self) -> int:
        return self.epochs_done_

    def _prediction_features(self, data) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("the model must be fitted before predict")
        frame = pd.DataFrame(data)
        if self.x is not None and set(self.x).issubset(frame.columns):
            frame = frame[self.x]
        array = frame.to_numpy(dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self.input_size:
            raise ValueError(f"expected {self.input_size} predictor columns")
        return torch.from_numpy(array)


class TorchClaMlp(_TorchMlpMixin, Classification):
    """Configurable PyTorch multilayer perceptron classifier.

    References:
        Goodfellow, I., Bengio, Y., and Courville, A. (2016). Deep Learning.
        MIT Press, chapter 6.
        Paszke, A., et al. (2019). PyTorch: An Imperative Style,
        High-Performance Deep Learning Library. NeurIPS 32.

    Porting:
        Mirrors ``torch_cla_mlp.R`` and replaces
        ``inst/python/torch_cla_mlp.py``. Fidelity: architecture, optimizer,
        validation and stopping rules match; CPU is the reproducible default.
    """

    def __init__(
        self,
        attribute: str,
        slevels: Sequence,
        input_size: int,
        hidden_sizes,
        num_classes: int | None = None,
        dropout: float = 0.5,
        activation: str = "relu",
        normalization: str = "none",
        init_method: str = "default",
        epochs: int = 100,
        learning_rate: float = 0.001,
        lr: float | None = None,
        validation_strategy: str = "static",
        stopping_rule: str = "none",
        negative_slope: float = 0.01,
        val_ratio: float = 0.33,
        batch_size: int = 1,
        patience: int = 3,
        min_delta: float = 0.0,
        sma_window: int = 30,
        ema_alpha: float = 0.2,
        test_window: int = 30,
        p_value: float = 0.05,
        weight_decay: float = 0.0,
        reshuffle_freq: int = 1,
        random_state: int | None = None,
        device: str | None = None,
    ) -> None:
        Classification.__init__(self, attribute, slevels)
        class_count = len(self.slevels) if num_classes is None else int(num_classes)
        if class_count != len(self.slevels):
            raise ValueError("num_classes must match the number of slevels")
        self.num_classes = class_count
        self.classes_ = list(self.slevels)
        self._initialize_mlp(
            input_size=input_size, hidden_sizes=hidden_sizes, output_size=class_count,
            dropout=dropout, activation_name=activation, output_activation="none",
            normalization=normalization, init_method=init_method, epochs=epochs,
            learning_rate=learning_rate, lr=lr, validation_strategy=validation_strategy,
            stopping_rule=stopping_rule, negative_slope=negative_slope,
            val_ratio=val_ratio, batch_size=batch_size, patience=patience,
            min_delta=min_delta, sma_window=sma_window, ema_alpha=ema_alpha,
            test_window=test_window, p_value=p_value, weight_decay=weight_decay,
            reshuffle_freq=reshuffle_freq, random_state=random_state, device=device,
        )

    def fit(self, data, **kwargs) -> TorchClaMlp:
        frame = self._prepare_classification_data(data)
        if len(self.x) != self.input_size:
            raise ValueError(
                f"input_size is {self.input_size}, but fitted data has {len(self.x)} predictors"
            )
        features = torch.from_numpy(frame[self.x].to_numpy(dtype=np.float32))
        mapping = {label: index for index, label in enumerate(self.slevels)}
        targets = torch.tensor(
            [mapping[value] for value in frame[self.attribute].astype(object)],
            dtype=torch.long,
        )
        self._train(features, targets, nn.CrossEntropyLoss())
        return self

    def predict(self, data, **kwargs) -> pd.DataFrame:
        features = self._prediction_features(data).to(self.device_)
        self.model.eval()
        with torch.no_grad():
            scores = functional.softmax(self.model(features), dim=-1).cpu().numpy()
        return pd.DataFrame(scores, columns=[str(level) for level in self.slevels])


class TorchRegMlp(_TorchMlpMixin, Regression):
    """Configurable PyTorch multilayer perceptron regressor.

    References:
        Goodfellow, I., Bengio, Y., and Courville, A. (2016). Deep Learning.
        MIT Press, chapter 6.
        Paszke, A., et al. (2019). PyTorch: An Imperative Style,
        High-Performance Deep Learning Library. NeurIPS 32.

    Porting:
        Mirrors ``torch_reg_mlp.R`` and replaces
        ``inst/python/torch_reg_mlp.py``. Fidelity: architecture, optimizer,
        validation and stopping rules match; CPU is the reproducible default.
    """

    def __init__(
        self,
        attribute: str,
        hidden_sizes,
        input_size: int | None = None,
        dropout: float = 0.5,
        activation: str = "relu",
        output_activation: str = "none",
        normalization: str = "none",
        init_method: str = "default",
        epochs: int = 100,
        learning_rate: float = 0.001,
        lr: float | None = None,
        validation_strategy: str = "static",
        stopping_rule: str = "none",
        negative_slope: float = 0.01,
        val_ratio: float = 0.33,
        batch_size: int = 1,
        patience: int = 3,
        min_delta: float = 0.0,
        sma_window: int = 30,
        ema_alpha: float = 0.2,
        test_window: int = 30,
        p_value: float = 0.05,
        reshuffle_freq: int = 1,
        random_state: int | None = None,
        device: str | None = None,
    ) -> None:
        Regression.__init__(self, attribute)
        self._initialize_mlp(
            input_size=input_size, hidden_sizes=hidden_sizes, output_size=1,
            dropout=dropout, activation_name=activation,
            output_activation=output_activation, normalization=normalization,
            init_method=init_method, epochs=epochs, learning_rate=learning_rate,
            lr=lr, validation_strategy=validation_strategy, stopping_rule=stopping_rule,
            negative_slope=negative_slope, val_ratio=val_ratio, batch_size=batch_size,
            patience=patience, min_delta=min_delta, sma_window=sma_window,
            ema_alpha=ema_alpha, test_window=test_window, p_value=p_value,
            weight_decay=0.0, reshuffle_freq=reshuffle_freq,
            random_state=random_state, device=device,
        )

    def fit(self, data, **kwargs) -> TorchRegMlp:
        frame = pd.DataFrame(data)
        if self.attribute not in frame.columns:
            raise ValueError(f"attribute '{self.attribute}' not found in data")
        frame = self._prepare_fit(frame)
        inferred = len(self.x)
        if self.input_size is None:
            self.input_size = inferred
        elif self.input_size != inferred:
            raise ValueError(
                f"input_size is {self.input_size}, but fitted data has {inferred} predictors"
            )
        features = torch.from_numpy(frame[self.x].to_numpy(dtype=np.float32))
        targets = torch.from_numpy(
            frame[self.attribute].to_numpy(dtype=np.float32).reshape(-1, 1)
        )
        self._train(features, targets, nn.MSELoss())
        return self

    def predict(self, data, **kwargs) -> np.ndarray:
        features = self._prediction_features(data).to(self.device_)
        self.model.eval()
        with torch.no_grad():
            return self.model(features).squeeze(-1).cpu().numpy()
