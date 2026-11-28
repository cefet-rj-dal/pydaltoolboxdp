"""Deep ++ neural sliding-window time-series forecasters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pytspredit import TsRegSW
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ._training import (
    StopController,
    TrainingConfig,
    activation,
    int_list,
    validate_training_options,
)
from .supervised import _MlpNetwork

__all__ = ["TorchTsMlp", "TsConv1D", "TsLstm"]


class _TorchTsMixin:
    """Shared legacy-compatible training loop for neural TsRegSW backends."""

    _shuffle_without_validation = False

    def _initialize_training(
        self,
        *,
        epochs: int,
        learning_rate: float,
        lr: float | None,
        validation_strategy: str,
        stopping_rule: str,
        val_ratio: float,
        batch_size: int,
        patience: int,
        min_delta: float,
        sma_window: int,
        ema_alpha: float,
        test_window: int,
        p_value: float,
        reshuffle_freq: int,
        random_state: int | None,
        device: str | None,
    ) -> None:
        validation_strategy, stopping_rule = validate_training_options(
            validation_strategy, stopping_rule
        )
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if not 0 < float(val_ratio) < 1:
            raise ValueError("val_ratio must be between 0 and 1")
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate if lr is None else lr)
        self.lr = self.learning_rate
        self.validation_strategy = validation_strategy
        self.stopping_rule = stopping_rule
        self.val_ratio = float(val_ratio)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.sma_window = int(sma_window)
        self.ema_alpha = float(ema_alpha)
        self.test_window = int(test_window)
        self.p_value = float(p_value)
        self.reshuffle_freq = max(1, int(reshuffle_freq))
        self.random_state = random_state
        self.device = device or "cpu"
        self.device_ = torch.device(self.device)
        self.network: nn.Module | None = None
        self.train_loss_: list[float] = []
        self.val_loss_: list[float] = []
        self.epochs_done_ = 0

    def _seed(self) -> None:
        if self.random_state is not None:
            np.random.seed(int(self.random_state))
            torch.manual_seed(int(self.random_state))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(self.random_state))

    def _build_network(self, input_size: int) -> nn.Module:  # pragma: no cover
        raise NotImplementedError

    def _reshape_features(self, values: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(values)

    def _loader(self, features, targets, indices=None, *, shuffle: bool) -> DataLoader:
        if indices is not None:
            features, targets = features[indices], targets[indices]
        return DataLoader(
            TensorDataset(features, targets),
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    def _static_split(self, length: int) -> tuple[np.ndarray, np.ndarray]:
        if length < 2:
            raise ValueError("validation requires at least two observations")
        validation_size = min(length - 1, max(1, int(length * self.val_ratio)))
        indices = np.arange(length)
        return indices[:-validation_size], indices[-validation_size:]

    def _dynamic_split(self, length: int) -> tuple[np.ndarray, np.ndarray]:
        if length < 2:
            raise ValueError("validation requires at least two observations")
        indices = np.arange(length)
        np.random.shuffle(indices)
        validation_size = min(length - 1, max(1, int(length * self.val_ratio)))
        return indices[validation_size:], indices[:validation_size]

    def _epoch(self, loader: DataLoader, optimizer=None) -> float:
        criterion = nn.MSELoss()
        losses = []
        self.network.train(optimizer is not None)
        context = torch.enable_grad() if optimizer is not None else torch.no_grad()
        with context:
            for features, targets in loader:
                features = features.float().to(self.device_)
                targets = targets.float().to(self.device_)
                if optimizer is not None:
                    optimizer.zero_grad()
                loss = criterion(self.network(features), targets)
                if optimizer is not None:
                    loss.backward()
                    optimizer.step()
                losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else 0.0

    def _do_fit(self, x, y):
        frame = pd.DataFrame(x)
        values = frame.to_numpy(dtype=np.float32)
        if values.ndim != 2 or values.shape[1] == 0:
            raise ValueError("time-series predictors must be a non-empty matrix")
        if self.input_size is None:
            self.input_size = values.shape[1]
        if values.shape[1] != int(self.input_size):
            raise ValueError(
                f"expected {self.input_size} lagged inputs, received {values.shape[1]}"
            )
        self._seed()
        self.device_ = torch.device(self.device)
        self.network = self._build_network(values.shape[1]).float().to(self.device_)
        self.model = self.network
        features = self._reshape_features(values)
        targets = torch.from_numpy(np.asarray(y, dtype=np.float32).reshape(-1, 1))
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
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
            train_indices, validation_indices = self._static_split(len(features))
            train_loader = self._loader(features, targets, train_indices, shuffle=True)
            validation_loader = self._loader(
                features, targets, validation_indices, shuffle=False
            )
        elif self.validation_strategy == "static":
            train_loader = self._loader(
                features, targets, shuffle=self._shuffle_without_validation
            )

        for epoch in range(self.epochs):
            self.epochs_done_ += 1
            if self.validation_strategy == "dynamic" and (
                epoch == 0 or self.epochs_done_ % self.reshuffle_freq == 0
            ):
                train_indices, validation_indices = self._dynamic_split(len(features))
                train_loader = self._loader(features, targets, train_indices, shuffle=True)
                validation_loader = self._loader(
                    features, targets, validation_indices, shuffle=False
                )
            self.train_loss_.append(self._epoch(train_loader, optimizer))
            if validation_loader is not None:
                value = self._epoch(validation_loader)
                self.val_loss_.append(value)
                if stopper.step(self.network, value):
                    break
        if stopper.best_state is not None:
            self.network.load_state_dict(stopper.best_state)
        return self

    def _do_predict(self, x) -> np.ndarray:
        if self.network is None:
            raise RuntimeError("the forecaster must be fitted before predict")
        values = pd.DataFrame(x).to_numpy(dtype=np.float32)
        features = self._reshape_features(values)
        loader = self._loader(
            features, torch.zeros((len(features), 1)), shuffle=False
        )
        predictions = []
        self.network.eval()
        with torch.no_grad():
            for batch, _ in loader:
                predictions.append(self.network(batch.float().to(self.device_)).cpu())
        return torch.vstack(predictions).squeeze(-1).numpy()

    @property
    def train_loss_hist(self) -> list[float]:
        return self.train_loss_

    @property
    def val_loss_hist(self) -> list[float]:
        return self.val_loss_

    @property
    def epochs_done(self) -> int:
        return self.epochs_done_


class TorchTsMlp(_TorchTsMixin, TsRegSW):
    """Feedforward neural forecaster over selected lagged inputs.

    References:
        Zhang, G., Patuwo, B. E., and Hu, M. Y. (1998). Forecasting with
        artificial neural networks: The state of the art. International Journal
        of Forecasting, 14(1):35-62.
        Paszke, A., et al. (2019). PyTorch. NeurIPS 32.

    Porting:
        Mirrors ``torch_ts_mlp.R`` and replaces ``inst/python/torch_ts_mlp.py``.
        Fidelity: exact architecture and training loop; ``TsRegSW`` supplies lag
        mapping, normalization, one-step and recursive multi-step forecasting.
    """

    _shuffle_without_validation = True

    def __init__(
        self,
        preprocess=None,
        input_size: int | None = None,
        input_map=None,
        hidden_sizes=(16, 8),
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
        TsRegSW.__init__(self, preprocess, input_size, input_map)
        self.hidden_sizes = int_list(hidden_sizes)
        self.dropout = float(dropout)
        self.activation = str(activation)
        self.output_activation = str(output_activation)
        self.normalization = str(normalization)
        self.init_method = str(init_method)
        self.negative_slope = float(negative_slope)
        self._initialize_training(
            epochs=epochs, learning_rate=learning_rate, lr=lr,
            validation_strategy=validation_strategy, stopping_rule=stopping_rule,
            val_ratio=val_ratio, batch_size=batch_size, patience=patience,
            min_delta=min_delta, sma_window=sma_window, ema_alpha=ema_alpha,
            test_window=test_window, p_value=p_value, reshuffle_freq=reshuffle_freq,
            random_state=random_state, device=device,
        )

    def _build_network(self, input_size: int) -> nn.Module:
        return _MlpNetwork(
            input_size, self.hidden_sizes, 1, dropout=self.dropout,
            hidden_activation=self.activation, output_activation=self.output_activation,
            normalization=self.normalization, init_method=self.init_method,
            negative_slope=self.negative_slope,
        )


class _Conv1DNetwork(nn.Module):
    def __init__(
        self, in_channels: int, sequence_length: int, conv_channels, kernel_sizes,
        strides, pooling: str, pool_kernel_size: int, dense_hidden_sizes,
        activation_name: str, negative_slope: float,
    ) -> None:
        super().__init__()
        layers = []
        previous = int(in_channels)
        for channels, kernel, stride in zip(conv_channels, kernel_sizes, strides):
            layers.append(nn.Conv1d(previous, channels, kernel_size=kernel, stride=stride))
            layers.append(activation(activation_name, negative_slope))
            if pooling == "max":
                layers.append(nn.MaxPool1d(pool_kernel_size))
            elif pooling == "avg":
                layers.append(nn.AvgPool1d(pool_kernel_size))
            previous = channels
        self.feature_extractor = nn.Sequential(*layers)
        with torch.no_grad():
            features = self.feature_extractor(
                torch.rand(1, int(in_channels), int(sequence_length))
            )
        head: list[nn.Module] = [nn.Flatten(1, -1)]
        previous = int(np.prod(features.shape[1:]))
        for size in dense_hidden_sizes:
            head.extend([nn.Linear(previous, size), activation(activation_name, negative_slope)])
            previous = size
        head.append(nn.Linear(previous, 1))
        self.regressor = nn.Sequential(*head)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.feature_extractor(values))


class TsConv1D(_TorchTsMixin, TsRegSW):
    """One-dimensional convolutional sliding-window forecaster.

    References:
        Bai, S., Kolter, J. Z., and Koltun, V. (2018). An Empirical Evaluation
        of Generic Convolutional and Recurrent Networks for Sequence Modeling.
        arXiv:1803.01271.
        Paszke, A., et al. (2019). PyTorch. NeurIPS 32.

    Porting:
        Mirrors ``ts_conv1d.R`` and replaces ``inst/python/ts_conv1d.py``.
        Fidelity: exact convolutional reshape, dense head and training loop;
        orchestration is inherited from ``TsRegSW``.
    """

    def __init__(
        self, preprocess=None, input_size: int | None = None, input_map=None,
        in_channels: int = 1, sequence_length: int | None = None,
        conv_channels=64, kernel_sizes=None, strides=1, pooling: str = "none",
        pool_kernel_size: int = 2, dense_hidden_sizes=50,
        activation: str = "relu", epochs: int = 100,
        learning_rate: float = 0.001, lr: float | None = None,
        validation_strategy: str = "static", stopping_rule: str = "none",
        negative_slope: float = 0.01, val_ratio: float = 0.33,
        batch_size: int = 1, patience: int = 3, min_delta: float = 0.0,
        sma_window: int = 30, ema_alpha: float = 0.2, test_window: int = 30,
        p_value: float = 0.05, reshuffle_freq: int = 1,
        random_state: int | None = None, device: str | None = None,
    ) -> None:
        TsRegSW.__init__(self, preprocess, input_size, input_map)
        if int(in_channels) <= 0:
            raise ValueError("in_channels must be positive")
        pooling = str(pooling).lower()
        if pooling not in {"none", "max", "avg"}:
            raise ValueError("pooling must be one of {'none', 'max', 'avg'}")
        self.in_channels = int(in_channels)
        self.sequence_length = None if sequence_length is None else int(sequence_length)
        self.conv_channels = int_list(conv_channels, default=[64])
        default_kernel = None
        if kernel_sizes is not None:
            default_kernel = int_list(kernel_sizes)
        self.kernel_sizes = default_kernel
        self.strides = int_list(strides, default=[1])
        self.pooling = pooling
        self.pool_kernel_size = int(pool_kernel_size)
        self.dense_hidden_sizes = int_list(dense_hidden_sizes, default=[50])
        self.activation = str(activation)
        self.negative_slope = float(negative_slope)
        self._initialize_training(
            epochs=epochs, learning_rate=learning_rate, lr=lr,
            validation_strategy=validation_strategy, stopping_rule=stopping_rule,
            val_ratio=val_ratio, batch_size=batch_size, patience=patience,
            min_delta=min_delta, sma_window=sma_window, ema_alpha=ema_alpha,
            test_window=test_window, p_value=p_value, reshuffle_freq=reshuffle_freq,
            random_state=random_state, device=device,
        )

    def _shape(self, input_size: int) -> tuple[int, int]:
        if input_size % self.in_channels != 0:
            raise ValueError("input_size must be divisible by in_channels")
        inferred = input_size // self.in_channels
        length = inferred if self.sequence_length is None else self.sequence_length
        if length * self.in_channels != input_size:
            raise ValueError("input_size must equal in_channels * sequence_length")
        return self.in_channels, length

    def _build_network(self, input_size: int) -> nn.Module:
        channels, length = self._shape(input_size)
        kernels = self.kernel_sizes or [1 if length == 1 else 2] * len(self.conv_channels)
        if not (len(self.conv_channels) == len(kernels) == len(self.strides)):
            raise ValueError("conv_channels, kernel_sizes and strides must have the same length")
        return _Conv1DNetwork(
            channels, length, self.conv_channels, kernels, self.strides, self.pooling,
            self.pool_kernel_size, self.dense_hidden_sizes, self.activation,
            self.negative_slope,
        )

    def _reshape_features(self, values: np.ndarray) -> torch.Tensor:
        channels, length = self._shape(values.shape[1])
        return torch.from_numpy(values).reshape(len(values), channels, length)


class _LstmNetwork(nn.Module):
    def __init__(
        self, feature_size: int, hidden_size: int, num_layers: int, dropout: float,
        bidirectional: bool, mlp_hidden_sizes, activation_name: str,
        negative_slope: float,
    ) -> None:
        super().__init__()
        effective_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=feature_size, hidden_size=hidden_size, num_layers=num_layers,
            dropout=effective_dropout, bidirectional=bidirectional, batch_first=True,
        )
        previous = hidden_size * (2 if bidirectional else 1)
        head = []
        for size in mlp_hidden_sizes:
            head.extend([nn.Linear(previous, size), activation(activation_name, negative_slope)])
            previous = size
        head.append(nn.Linear(previous, 1))
        self.fc = nn.Sequential(*head)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(values)
        return self.fc(output[:, -1, :])


class TsLstm(_TorchTsMixin, TsRegSW):
    """LSTM sliding-window forecaster with an optional dense head.

    References:
        Hochreiter, S., and Schmidhuber, J. (1997). Long Short-Term Memory.
        Neural Computation, 9(8):1735-1780.
        Paszke, A., et al. (2019). PyTorch. NeurIPS 32.

    Porting:
        Mirrors ``ts_lstm.R`` and replaces ``inst/python/ts_lstm.py``.
        Fidelity: exact sequence reshape, recurrent stack, dense head and
        training loop; one-step/multi-step behavior comes from ``TsRegSW``.
    """

    def __init__(
        self, preprocess=None, input_size: int | None = None, input_map=None,
        hidden_size: int | None = None, sequence_length: int | None = None,
        num_layers: int = 1, dropout: float = 0.0, bidirectional: bool = False,
        mlp_hidden_sizes=(), activation: str = "relu", epochs: int = 100,
        learning_rate: float = 0.001, lr: float | None = None,
        validation_strategy: str = "static", stopping_rule: str = "none",
        negative_slope: float = 0.01, val_ratio: float = 0.33,
        batch_size: int = 1, patience: int = 3, min_delta: float = 0.0,
        sma_window: int = 30, ema_alpha: float = 0.2, test_window: int = 30,
        p_value: float = 0.05, reshuffle_freq: int = 1,
        random_state: int | None = None, device: str | None = None,
    ) -> None:
        TsRegSW.__init__(self, preprocess, input_size, input_map)
        self.hidden_size = None if hidden_size is None else int(hidden_size)
        self.sequence_length = None if sequence_length is None else int(sequence_length)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.bidirectional = bool(bidirectional)
        self.mlp_hidden_sizes = int_list(mlp_hidden_sizes, allow_empty=True)
        self.activation = str(activation)
        self.negative_slope = float(negative_slope)
        self._initialize_training(
            epochs=epochs, learning_rate=learning_rate, lr=lr,
            validation_strategy=validation_strategy, stopping_rule=stopping_rule,
            val_ratio=val_ratio, batch_size=batch_size, patience=patience,
            min_delta=min_delta, sma_window=sma_window, ema_alpha=ema_alpha,
            test_window=test_window, p_value=p_value, reshuffle_freq=reshuffle_freq,
            random_state=random_state, device=device,
        )

    def _shape(self, input_size: int) -> tuple[int, int]:
        length = input_size if self.sequence_length is None else self.sequence_length
        if length <= 0 or input_size % length != 0:
            raise ValueError("input_size must be divisible by sequence_length")
        return length, input_size // length

    def _build_network(self, input_size: int) -> nn.Module:
        _, feature_size = self._shape(input_size)
        hidden_size = input_size if self.hidden_size is None else self.hidden_size
        return _LstmNetwork(
            feature_size, hidden_size, self.num_layers, self.dropout,
            self.bidirectional, self.mlp_hidden_sizes, self.activation,
            self.negative_slope,
        )

    def _reshape_features(self, values: np.ndarray) -> torch.Tensor:
        length, feature_size = self._shape(values.shape[1])
        return torch.from_numpy(values).reshape(len(values), length, feature_size)
