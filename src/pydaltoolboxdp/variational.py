"""Variational Deep ++ autoencoders."""

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
    activation,
    dense_stack,
    int_list,
    validate_training_options,
)

__all__ = ["AutoencVariationalE", "AutoencVariationalEd"]


class _VariationalNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        encoding_size: int,
        encoder_hidden_sizes,
        decoder_hidden_sizes,
        activation_name: str,
        negative_slope: float,
        output_activation: str,
    ) -> None:
        super().__init__()
        encoder_hidden_sizes = int_list(
            encoder_hidden_sizes, default=[64, 32]
        )
        decoder_hidden_sizes = int_list(
            decoder_hidden_sizes, default=list(reversed(encoder_hidden_sizes))
        )
        layers: list[nn.Module] = []
        previous = input_size
        for hidden_size in encoder_hidden_sizes:
            layers.append(nn.Linear(previous, hidden_size))
            layers.append(activation(activation_name, negative_slope))
            previous = hidden_size
        self.encoder = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(previous, encoding_size)
        self.var_layer = nn.Linear(previous, encoding_size)
        self.decoder = dense_stack(
            encoding_size,
            decoder_hidden_sizes,
            input_size,
            hidden_activation=activation_name,
            output_activation=output_activation,
            negative_slope=negative_slope,
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        return self.mean_layer(hidden), self.var_layer(hidden)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    @staticmethod
    def reparameterization(mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        return mean + var * torch.randn_like(var)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, var = self.encode(x)
        return self.decode(self.reparameterization(mean, var)), mean, var


class _VariationalAutoencoderMixin:
    def _initialize(
        self,
        *,
        input_size: int,
        encoding_size: int,
        encoder_hidden_sizes=(64, 32),
        decoder_hidden_sizes=None,
        activation: str = "leaky_relu",
        output_activation: str = "sigmoid",
        reconstruction_loss: str = "bce",
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
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        reconstruction_loss = str(reconstruction_loss).lower()
        if reconstruction_loss not in {"bce", "mse"}:
            raise ValueError("reconstruction_loss must be one of {'bce', 'mse'}")
        validation_strategy, stopping_rule = validate_training_options(
            validation_strategy, stopping_rule
        )

        self.input_size = int(input_size)
        self.encoding_size = int(encoding_size)
        self.encoder_hidden_sizes = int_list(encoder_hidden_sizes, default=[64, 32])
        self.decoder_hidden_sizes = (
            list(reversed(self.encoder_hidden_sizes))
            if decoder_hidden_sizes is None
            else int_list(decoder_hidden_sizes)
        )
        self.activation = str(activation)
        self.output_activation = str(output_activation)
        self.reconstruction_loss = reconstruction_loss
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
        self.model = None
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
        self.model = _VariationalNetwork(
            self.input_size,
            self.encoding_size,
            self.encoder_hidden_sizes,
            self.decoder_hidden_sizes,
            self.activation,
            self.negative_slope,
            self.output_activation,
        ).float().to(self.device_)

    def _array(self, data) -> np.ndarray:
        array = data.to_numpy() if isinstance(data, pd.DataFrame) else np.asarray(data)
        array = np.asarray(array, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self.input_size:
            raise ValueError(f"expected a 2D array with {self.input_size} features")
        if len(array) == 0 or not np.isfinite(array).all():
            raise ValueError("autoencoder data must be non-empty and finite")
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
        indices = np.arange(length)
        np.random.shuffle(indices)
        validation_size = min(length - 1, max(1, int(length * self.val_ratio)))
        return indices[validation_size:], indices[:validation_size]

    def _loss(self, outputs, inputs, mean, var) -> torch.Tensor:
        if self.reconstruction_loss == "bce":
            reconstruction = nn.functional.binary_cross_entropy(
                outputs, inputs, reduction="sum"
            )
        else:
            reconstruction = nn.functional.mse_loss(outputs, inputs, reduction="sum")
        divergence = -0.5 * torch.sum(1 + var - mean.pow(2) - var.exp())
        return reconstruction + divergence

    def _run_epoch(self, loader: DataLoader, optimizer=None) -> float:
        losses: list[float] = []
        self.model.train(optimizer is not None)
        context = torch.enable_grad() if optimizer is not None else torch.no_grad()
        with context:
            for features, targets in loader:
                features = features.to(self.device_)
                targets = targets.to(self.device_)
                if optimizer is not None:
                    optimizer.zero_grad()
                output, mean, var = self.model(features)
                loss = self._loss(output, targets, mean, var)
                if optimizer is not None:
                    loss.backward()
                    optimizer.step()
                losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else 0.0

    def fit(self, data, **kwargs):
        for name in (
            "batch_size", "epochs", "learning_rate", "validation_strategy",
            "stopping_rule", "val_ratio", "patience", "min_delta", "sma_window",
            "ema_alpha", "test_window", "p_value", "reshuffle_freq",
        ):
            if name in kwargs and kwargs[name] is not None:
                setattr(self, name, kwargs[name])
        if kwargs.get("num_epochs") is not None:
            self.epochs = int(kwargs["num_epochs"])
        self.num_epochs = int(self.epochs)
        self.validation_strategy, self.stopping_rule = validate_training_options(
            self.validation_strategy, self.stopping_rule
        )
        array = self._array(data)
        self._build_model()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.learning_rate))
        config = TrainingConfig(
            batch_size=int(self.batch_size), epochs=int(self.epochs),
            learning_rate=float(self.learning_rate),
            validation_strategy=self.validation_strategy, stopping_rule=self.stopping_rule,
            val_ratio=float(self.val_ratio), patience=int(self.patience),
            min_delta=float(self.min_delta), sma_window=int(self.sma_window),
            ema_alpha=float(self.ema_alpha), test_window=int(self.test_window),
            p_value=float(self.p_value), reshuffle_freq=max(1, int(self.reshuffle_freq)),
        )
        stopper = StopController(config)
        self.train_loss_, self.val_loss_, self.epochs_done_ = [], [], 0
        train_loader = validation_loader = None
        if self.validation_strategy == "static" and self.stopping_rule != "none":
            train_indices, validation_indices = self._split(len(array))
            train_loader = self._loader(array[train_indices], shuffle=True)
            validation_loader = self._loader(array[validation_indices], shuffle=False)
        elif self.validation_strategy == "static":
            train_loader = self._loader(array, shuffle=True)
        for epoch in range(config.epochs):
            self.epochs_done_ += 1
            if self.validation_strategy == "dynamic" and (
                epoch == 0 or self.epochs_done_ % config.reshuffle_freq == 0
            ):
                train_indices, validation_indices = self._split(len(array))
                train_loader = self._loader(array[train_indices], shuffle=True)
                validation_loader = self._loader(array[validation_indices], shuffle=False)
            self.train_loss_.append(self._run_epoch(train_loader, optimizer))
            if validation_loader is not None:
                validation_loss = self._run_epoch(validation_loader)
                self.val_loss_.append(validation_loss)
                if stopper.step(self.model, validation_loss):
                    break
        if stopper.best_state is not None:
            self.model.load_state_dict(stopper.best_state)
        return self

    @property
    def train_loss(self):
        return self.train_loss_

    @property
    def val_loss(self):
        return self.val_loss_

    def _transform(self, data, *, reconstruct: bool) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("the autoencoder must be fitted before transform")
        loader = self._loader(self._array(data), shuffle=False)
        result = []
        self.model.eval()
        with torch.no_grad():
            for features, _ in loader:
                decoded, mean, var = self.model(features.to(self.device_))
                output = decoded if reconstruct else torch.cat([mean, var], dim=1)
                result.append(output.cpu().numpy())
        return np.concatenate(result, axis=0)


class AutoencVariationalE(_VariationalAutoencoderMixin, AutoencBaseE):
    """Variational encoder returning latent mean and variance parameters.

    References:
        Kingma, D. P., and Welling, M. (2014). Auto-Encoding Variational Bayes. ICLR.

    Porting:
        Mirrors ``autoenc_variational_e.R`` and replaces
        ``inst/python/autoenc_variational.py``. Fidelity: stochastic
        reparameterization, reconstruction/KL loss and exported mean/variance match
        the legacy backend.
    """

    def __init__(self, input_size: int, encoding_size: int, **kwargs) -> None:
        AutoencBaseE.__init__(self, input_size, encoding_size)
        self._initialize(input_size=input_size, encoding_size=encoding_size, **kwargs)

    def transform(self, data, **kwargs) -> np.ndarray:
        return self._transform(data, reconstruct=False)


class AutoencVariationalEd(_VariationalAutoencoderMixin, AutoencBaseEd):
    """Variational encoder-decoder returning stochastic reconstructions.

    References:
        Kingma, D. P., and Welling, M. (2014). Auto-Encoding Variational Bayes. ICLR.

    Porting:
        Mirrors ``autoenc_variational_ed.R`` and replaces
        ``inst/python/autoenc_variational.py``. Fidelity: stochastic
        reparameterization, reconstruction/KL loss and decoder output match the
        legacy backend.
    """

    def __init__(self, input_size: int, encoding_size: int, **kwargs) -> None:
        AutoencBaseEd.__init__(self, input_size, encoding_size)
        self._initialize(input_size=input_size, encoding_size=encoding_size, **kwargs)

    def transform(self, data, **kwargs) -> np.ndarray:
        return self._transform(data, reconstruct=True)
