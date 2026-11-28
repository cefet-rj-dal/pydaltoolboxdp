"""Greedy layer-wise stacked Deep ++ autoencoders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pydaltoolbox import AutoencBaseE, AutoencBaseEd
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ._training import (
    StopController,
    TrainingConfig,
    dense_stack,
    int_list,
    validate_training_options,
)

__all__ = ["AutoencStackedE", "AutoencStackedEd"]


class _StackUnit(nn.Module):
    def __init__(
        self,
        input_size: int,
        encoding_size: int,
        encoder_hidden_sizes,
        decoder_hidden_sizes,
        activation: str,
        output_activation: str,
        negative_slope: float,
    ) -> None:
        super().__init__()
        encoder_hidden_sizes = (
            [64] if encoder_hidden_sizes is None else encoder_hidden_sizes
        )
        decoder_hidden_sizes = (
            list(reversed(list(encoder_hidden_sizes)))
            if decoder_hidden_sizes is None
            else decoder_hidden_sizes
        )
        self.encoder = dense_stack(
            input_size,
            encoder_hidden_sizes,
            encoding_size,
            hidden_activation=activation,
            negative_slope=negative_slope,
        )
        self.decoder = dense_stack(
            encoding_size,
            decoder_hidden_sizes,
            input_size,
            hidden_activation=activation,
            output_activation=output_activation,
            negative_slope=negative_slope,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class _StackedAutoencoderMixin:
    def _initialize(
        self,
        *,
        input_size: int,
        encoding_size: int,
        k: int = 3,
        encoding_sizes=None,
        encoder_hidden_sizes=64,
        decoder_hidden_sizes=None,
        activation: str = "relu",
        output_activation: str = "none",
        batch_size: int = 1,
        epochs: int = 100,
        num_epochs: int | None = None,
        learning_rate: float = 0.001,
        validation_strategy: str = "static",
        stopping_rule: str = "none",
        negative_slope: float = 0.01,
        val_ratio: float = 0.33,
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
        if int(input_size) <= 0 or int(encoding_size) <= 0:
            raise ValueError("input_size and encoding_size must be positive")
        if int(k) <= 0:
            raise ValueError("k must be positive")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if not 0 < float(val_ratio) < 1:
            raise ValueError("val_ratio must be between 0 and 1")
        validation_strategy, stopping_rule = validate_training_options(
            validation_strategy, stopping_rule
        )

        self.input_size = int(input_size)
        self.encoding_size = int(encoding_size)
        self.k = int(k)
        self.encoding_sizes = int_list(
            encoding_sizes, default=[self.encoding_size] * self.k
        )
        self.stage_count = len(self.encoding_sizes)
        self.encoder_hidden_sizes = self._expand_stage_param(
            encoder_hidden_sizes, self.stage_count, default=[64]
        )
        self.decoder_hidden_sizes = self._expand_stage_param(
            decoder_hidden_sizes, self.stage_count, default=None
        )
        self.activation = str(activation)
        self.output_activation = str(output_activation)
        self.negative_slope = float(negative_slope)
        self.batch_size = int(batch_size)
        self.epochs = int(num_epochs if num_epochs is not None else epochs)
        self.num_epochs = self.epochs
        self.learning_rate = float(learning_rate)
        self.validation_strategy = validation_strategy
        self.stopping_rule = stopping_rule
        self.val_ratio = float(val_ratio)
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

        self.stack: list[_StackUnit] = []
        self.train_loss_: list[float] = []
        self.val_loss_: list[float] = []
        self.epochs_done_ = 0

    @staticmethod
    def _expand_stage_param(value, stage_count: int, default=None):
        if value is None:
            return [default for _ in range(stage_count)]
        if (
            isinstance(value, (list, tuple))
            and value
            and isinstance(value[0], (list, tuple, np.ndarray))
        ):
            if len(value) != stage_count:
                raise ValueError(
                    "stage-specific parameter lists must match the number of stages"
                )
            return [list(stage) if stage is not None else default for stage in value]
        return [value for _ in range(stage_count)]

    def _seed(self) -> None:
        if self.random_state is not None:
            np.random.seed(int(self.random_state))
            torch.manual_seed(int(self.random_state))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(self.random_state))

    def _build_stack(self) -> None:
        self._seed()
        self.device_ = torch.device(self.device)
        self.stack = []
        current_size = self.input_size
        for index, latent_size in enumerate(self.encoding_sizes):
            unit = _StackUnit(
                current_size,
                latent_size,
                self.encoder_hidden_sizes[index],
                self.decoder_hidden_sizes[index],
                self.activation,
                self.output_activation,
                self.negative_slope,
            ).float().to(self.device_)
            self.stack.append(unit)
            current_size = latent_size

    def _array(self, data) -> np.ndarray:
        array = data.to_numpy() if isinstance(data, pd.DataFrame) else np.asarray(data)
        array = np.asarray(array, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("autoencoder data must be a two-dimensional array")
        if array.shape[1] != self.input_size:
            raise ValueError(
                f"expected {self.input_size} features, received {array.shape[1]}"
            )
        if array.shape[0] == 0:
            raise ValueError("autoencoder data must contain at least one observation")
        if not np.isfinite(array).all():
            raise ValueError("autoencoder data contains non-finite values")
        return array

    def _loader(self, array: np.ndarray, *, shuffle: bool) -> DataLoader:
        tensor = torch.from_numpy(np.array(array, dtype=np.float32, copy=True))
        return DataLoader(
            TensorDataset(tensor, tensor),
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    def _split(self, length: int) -> tuple[np.ndarray, np.ndarray]:
        if length < 2:
            raise ValueError("validation requires at least two observations")
        indices = np.arange(length)
        np.random.shuffle(indices)
        validation_size = min(length - 1, max(1, int(length * self.val_ratio)))
        return indices[validation_size:], indices[:validation_size]

    def _run_epoch(self, unit: nn.Module, loader: DataLoader, optimizer=None) -> float:
        criterion = nn.MSELoss()
        losses: list[float] = []
        if optimizer is None:
            unit.eval()
            with torch.no_grad():
                for features, targets in loader:
                    features = features.to(self.device_)
                    targets = targets.to(self.device_)
                    losses.append(float(criterion(unit(features), targets).item()))
        else:
            unit.train()
            for features, targets in loader:
                features = features.to(self.device_)
                targets = targets.to(self.device_)
                optimizer.zero_grad()
                loss = criterion(unit(features), targets)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else 0.0

    def _fit_unit(
        self, unit: nn.Module, array: np.ndarray, config: TrainingConfig
    ) -> tuple[list[float], list[float], int]:
        optimizer = torch.optim.Adam(unit.parameters(), lr=config.learning_rate)
        stopper = StopController(config)
        train_history: list[float] = []
        validation_history: list[float] = []
        epochs_done = 0
        train_loader = None
        validation_loader = None

        if self.validation_strategy == "static" and self.stopping_rule != "none":
            train_indices, validation_indices = self._split(len(array))
            train_loader = self._loader(array[train_indices], shuffle=True)
            validation_loader = self._loader(array[validation_indices], shuffle=False)
        elif self.validation_strategy == "static":
            train_loader = self._loader(array, shuffle=True)

        for epoch in range(config.epochs):
            epochs_done += 1
            if self.validation_strategy == "dynamic" and (
                epoch == 0 or (epoch + 1) % config.reshuffle_freq == 0
            ):
                train_indices, validation_indices = self._split(len(array))
                train_loader = self._loader(array[train_indices], shuffle=True)
                validation_loader = self._loader(
                    array[validation_indices], shuffle=False
                )
            train_history.append(self._run_epoch(unit, train_loader, optimizer))
            if validation_loader is not None:
                validation_loss = self._run_epoch(unit, validation_loader)
                validation_history.append(validation_loss)
                if stopper.step(unit, validation_loss):
                    break

        if stopper.best_state is not None:
            unit.load_state_dict(stopper.best_state)
        return train_history, validation_history, epochs_done

    def _apply(self, module: nn.Module, array: np.ndarray) -> np.ndarray:
        loader = self._loader(array, shuffle=False)
        result = []
        module.eval()
        with torch.no_grad():
            for features, _ in loader:
                output = module(features.to(self.device_))
                result.append(output.cpu().numpy())
        return np.concatenate(result, axis=0)

    def fit(self, data, **kwargs):
        for name in (
            "batch_size",
            "epochs",
            "learning_rate",
            "validation_strategy",
            "stopping_rule",
            "val_ratio",
            "patience",
            "min_delta",
            "sma_window",
            "ema_alpha",
            "test_window",
            "p_value",
            "reshuffle_freq",
        ):
            if name in kwargs and kwargs[name] is not None:
                setattr(self, name, kwargs[name])
        if kwargs.get("num_epochs") is not None:
            self.epochs = int(kwargs["num_epochs"])
        self.num_epochs = int(self.epochs)
        self.validation_strategy, self.stopping_rule = validate_training_options(
            self.validation_strategy, self.stopping_rule
        )

        current = self._array(data)
        self._build_stack()
        config = TrainingConfig(
            batch_size=int(self.batch_size),
            epochs=int(self.epochs),
            learning_rate=float(self.learning_rate),
            validation_strategy=self.validation_strategy,
            stopping_rule=self.stopping_rule,
            val_ratio=float(self.val_ratio),
            patience=int(self.patience),
            min_delta=float(self.min_delta),
            sma_window=int(self.sma_window),
            ema_alpha=float(self.ema_alpha),
            test_window=int(self.test_window),
            p_value=float(self.p_value),
            reshuffle_freq=max(1, int(self.reshuffle_freq)),
        )
        self.train_loss_ = []
        self.val_loss_ = []
        self.epochs_done_ = 0
        for unit in self.stack:
            train_history, validation_history, epochs_done = self._fit_unit(
                unit, current, config
            )
            self.train_loss_ = train_history
            self.val_loss_ = validation_history
            self.epochs_done_ += epochs_done
            current = self._apply(unit.encoder, current)
        return self

    @property
    def train_loss(self) -> list[float]:
        return self.train_loss_

    @property
    def val_loss(self) -> list[float]:
        return self.val_loss_

    def _require_fitted(self) -> None:
        if not self.stack:
            raise RuntimeError("the autoencoder must be fitted before transform")

    def encode(self, data) -> np.ndarray:
        self._require_fitted()
        current = self._array(data)
        for unit in self.stack:
            current = self._apply(unit.encoder, current)
        return current

    def reconstruct(self, data) -> np.ndarray:
        self._require_fitted()
        current = self._array(data)
        for unit in self.stack:
            current = self._apply(unit.encoder, current)
        for unit in reversed(self.stack):
            current = self._apply(unit.decoder, current)
        return current


class AutoencStackedE(_StackedAutoencoderMixin, AutoencBaseE):
    """Greedy stacked encoder that learns hierarchical representations.

    References:
        Bengio, Y., Lamblin, P., Popovici, D., and Larochelle, H. (2007).
        Greedy Layer-Wise Training of Deep Networks. NeurIPS 19.

    Porting:
        Mirrors ``autoenc_stacked_e.R`` and replaces
        ``inst/python/autoenc_stacked.py``. Fidelity: stage construction,
        greedy layer-wise optimization, latent propagation and loss exposure match
        the legacy backend.
    """

    def __init__(self, input_size: int, encoding_size: int, **kwargs) -> None:
        AutoencBaseE.__init__(self, input_size, encoding_size)
        self._initialize(input_size=input_size, encoding_size=encoding_size, **kwargs)

    def transform(self, data, **kwargs) -> np.ndarray:
        return self.encode(data)


class AutoencStackedEd(_StackedAutoencoderMixin, AutoencBaseEd):
    """Greedy stacked encoder-decoder that reconstructs observations.

    References:
        Bengio, Y., Lamblin, P., Popovici, D., and Larochelle, H. (2007).
        Greedy Layer-Wise Training of Deep Networks. NeurIPS 19.

    Porting:
        Mirrors ``autoenc_stacked_ed.R`` and replaces
        ``inst/python/autoenc_stacked.py``. Fidelity: stage construction,
        greedy layer-wise optimization, reverse decoding and loss exposure match
        the legacy backend.
    """

    def __init__(self, input_size: int, encoding_size: int, **kwargs) -> None:
        AutoencBaseEd.__init__(self, input_size, encoding_size)
        self._initialize(input_size=input_size, encoding_size=encoding_size, **kwargs)

    def transform(self, data, **kwargs) -> np.ndarray:
        return self.reconstruct(data)
