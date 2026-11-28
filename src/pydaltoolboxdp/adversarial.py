"""Adversarial Deep ++ autoencoders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pydaltoolbox import AutoencBaseE, AutoencBaseEd
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset

from ._training import (
    StopController,
    TrainingConfig,
    int_list,
    validate_training_options,
)

__all__ = ["AutoencAdvE", "AutoencAdvEd"]


def _apply_activation(name: str, x: torch.Tensor, negative_slope: float) -> torch.Tensor:
    name = str(name).lower()
    functions = {
        "relu": functional.relu,
        "leaky_relu": lambda value: functional.leaky_relu(
            value, negative_slope=negative_slope
        ),
        "elu": functional.elu,
        "gelu": functional.gelu,
        "tanh": torch.tanh,
    }
    try:
        return functions[name](x)
    except KeyError as error:
        raise ValueError(f"unsupported activation: {name}") from error


class _FeedForwardNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes,
        output_size: int,
        activation: str,
        dropout: float,
        output_activation: str = "none",
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        previous = input_size
        for hidden_size in int_list(hidden_sizes, allow_empty=True):
            self.layers.append(nn.Linear(previous, hidden_size))
            previous = hidden_size
        self.output = nn.Linear(previous, output_size)
        self.activation = str(activation).lower()
        self.dropout = float(dropout)
        self.output_activation = str(output_activation).lower()
        self.negative_slope = float(negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = _apply_activation(self.activation, layer(x), self.negative_slope)
            if self.dropout > 0:
                x = functional.dropout(x, p=self.dropout, training=self.training)
        x = self.output(x)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        if self.output_activation == "tanh":
            return torch.tanh(x)
        return x


class _AdversarialAutoencoderMixin:
    def _initialize(
        self,
        *,
        input_size: int,
        encoding_size: int,
        encoder_hidden_sizes=(60, 60),
        decoder_hidden_sizes=(60, 60),
        discriminator_hidden_sizes=(60, 60),
        activation: str = "relu",
        dropout: float = 0.5,
        latent_prior_scale: float = 5.0,
        lr_encoder: float | None = None,
        lr_decoder: float | None = None,
        lr_generator: float | None = None,
        lr_discriminator: float | None = None,
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
        if not 0 <= float(dropout) < 1:
            raise ValueError("dropout must be in the interval [0, 1)")
        validation_strategy, stopping_rule = validate_training_options(
            validation_strategy, stopping_rule
        )
        self.input_size = int(input_size)
        self.encoding_size = int(encoding_size)
        self.encoder_hidden_sizes = int_list(encoder_hidden_sizes, default=[60, 60])
        self.decoder_hidden_sizes = int_list(decoder_hidden_sizes, default=[60, 60])
        self.discriminator_hidden_sizes = int_list(
            discriminator_hidden_sizes, default=[60, 60]
        )
        self.activation = str(activation).lower()
        self.dropout = float(dropout)
        self.latent_prior_scale = float(latent_prior_scale)
        self.lr_encoder = None if lr_encoder is None else float(lr_encoder)
        self.lr_decoder = None if lr_decoder is None else float(lr_decoder)
        self.lr_generator = None if lr_generator is None else float(lr_generator)
        self.lr_discriminator = (
            None if lr_discriminator is None else float(lr_discriminator)
        )
        self.batch_size = int(batch_size)
        self.epochs = int(num_epochs if num_epochs is not None else epochs)
        self.num_epochs = self.epochs
        self.learning_rate = float(learning_rate)
        self.validation_strategy = validation_strategy
        self.stopping_rule = stopping_rule
        self.negative_slope = float(negative_slope)
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
        self.Q = self.P = self.D_gauss = None
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
        self.Q = _FeedForwardNet(
            self.input_size, self.encoder_hidden_sizes, self.encoding_size,
            self.activation, self.dropout, negative_slope=self.negative_slope,
        ).to(self.device_)
        self.P = _FeedForwardNet(
            self.encoding_size, self.decoder_hidden_sizes, self.input_size,
            self.activation, self.dropout, output_activation="sigmoid",
            negative_slope=self.negative_slope,
        ).to(self.device_)
        self.D_gauss = _FeedForwardNet(
            self.encoding_size, self.discriminator_hidden_sizes, 1,
            self.activation, self.dropout, output_activation="sigmoid",
            negative_slope=self.negative_slope,
        ).to(self.device_)

    def _reset_optimizers(self) -> None:
        encoder_lr = 1e-4 if self.lr_encoder is None else self.lr_encoder
        decoder_lr = 1e-4 if self.lr_decoder is None else self.lr_decoder
        generator_lr = 5e-5 if self.lr_generator is None else self.lr_generator
        discriminator_lr = 5e-5 if self.lr_discriminator is None else self.lr_discriminator
        self.encoder_opt = torch.optim.Adam(self.Q.parameters(), lr=encoder_lr)
        self.decoder_opt = torch.optim.Adam(self.P.parameters(), lr=decoder_lr)
        self.generator_opt = torch.optim.Adam(self.Q.parameters(), lr=generator_lr)
        self.discriminator_opt = torch.optim.Adam(
            self.D_gauss.parameters(), lr=discriminator_lr
        )

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
            TensorDataset(tensor, tensor), batch_size=self.batch_size,
            shuffle=shuffle, drop_last=False,
        )

    def _split(self, length: int) -> tuple[np.ndarray, np.ndarray]:
        indices = np.arange(length)
        np.random.shuffle(indices)
        validation_size = min(length - 1, max(1, int(length * self.val_ratio)))
        return indices[validation_size:], indices[:validation_size]

    def _zero_all(self) -> None:
        self.P.zero_grad(); self.Q.zero_grad(); self.D_gauss.zero_grad()

    def _train_epoch(self, loader: DataLoader) -> float:
        tiny = 1e-15
        losses: list[float] = []
        self.Q.train(); self.P.train(); self.D_gauss.train()
        for features, _ in loader:
            features = features.to(self.device_).view(len(features), -1)
            latent = self.Q(features)
            reconstruction = self.P(latent)
            reconstruction_loss = nn.MSELoss()(reconstruction + tiny, features + tiny)
            reconstruction_loss.backward()
            self.decoder_opt.step(); self.encoder_opt.step(); self._zero_all()

            self.Q.eval()
            real = torch.randn(
                len(features), self.encoding_size, device=self.device_
            ) * self.latent_prior_scale
            fake = self.Q(features)
            real_score = self.D_gauss(real)
            fake_score = self.D_gauss(fake)
            discriminator_loss = -torch.mean(
                torch.log(real_score + tiny) + torch.log(1 - fake_score + tiny)
            )
            discriminator_loss.backward(); self.discriminator_opt.step(); self._zero_all()

            self.Q.train()
            fake_score = self.D_gauss(self.Q(features))
            generator_loss = -torch.mean(torch.log(fake_score + tiny))
            generator_loss.backward(); self.generator_opt.step(); self._zero_all()
            losses.append(float(reconstruction_loss.item()))
        return float(np.mean(losses)) if losses else 0.0

    def _eval_epoch(self, loader: DataLoader) -> float:
        losses: list[float] = []
        self.Q.eval(); self.P.eval()
        with torch.no_grad():
            for features, _ in loader:
                features = features.to(self.device_).view(len(features), -1)
                losses.append(float(nn.MSELoss()(self.P(self.Q(features)), features).item()))
        return float(np.mean(losses)) if losses else 0.0

    def state_dict(self) -> dict:
        return {"Q": self.Q.state_dict(), "P": self.P.state_dict(), "D": self.D_gauss.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.Q.load_state_dict(state["Q"]); self.P.load_state_dict(state["P"])
        self.D_gauss.load_state_dict(state["D"])

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
        self._build_model()
        if float(self.learning_rate) > 0:
            if self.lr_encoder is None: self.lr_encoder = float(self.learning_rate)
            if self.lr_decoder is None: self.lr_decoder = float(self.learning_rate)
            if self.lr_generator is None: self.lr_generator = float(self.learning_rate) * 0.5
            if self.lr_discriminator is None: self.lr_discriminator = float(self.learning_rate) * 0.5
        self._reset_optimizers()
        array = self._array(data)
        config = TrainingConfig(
            batch_size=int(self.batch_size), epochs=int(self.epochs),
            learning_rate=float(self.learning_rate), validation_strategy=self.validation_strategy,
            stopping_rule=self.stopping_rule, val_ratio=float(self.val_ratio),
            patience=int(self.patience), min_delta=float(self.min_delta),
            sma_window=int(self.sma_window), ema_alpha=float(self.ema_alpha),
            test_window=int(self.test_window), p_value=float(self.p_value),
            reshuffle_freq=max(1, int(self.reshuffle_freq)),
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
            self.train_loss_.append(self._train_epoch(train_loader))
            if validation_loader is not None:
                validation_loss = self._eval_epoch(validation_loader)
                self.val_loss_.append(validation_loss)
                if self.stopping_rule == "none":
                    stopper.best_state = StopController._clone_state(self)
                elif stopper.step(self, validation_loss):
                    break
        if stopper.best_state is not None:
            self.load_state_dict(stopper.best_state)
        return self

    @property
    def train_loss(self): return self.train_loss_

    @property
    def val_loss(self): return self.val_loss_

    def _transform(self, data, *, reconstruct: bool) -> np.ndarray:
        if self.Q is None:
            raise RuntimeError("the autoencoder must be fitted before transform")
        loader = self._loader(self._array(data), shuffle=False)
        result = []
        self.Q.eval(); self.P.eval()
        with torch.no_grad():
            for features, _ in loader:
                flat = features.to(self.device_).view(len(features), -1)
                output = self.P(self.Q(flat)) if reconstruct else self.Q(flat)
                result.append(output.cpu().numpy())
        return np.concatenate(result, axis=0)


class AutoencAdvE(_AdversarialAutoencoderMixin, AutoencBaseE):
    """Adversarial encoder regularized against a Gaussian latent prior.

    References:
        Makhzani, A., Shlens, J., Jaitly, N., Goodfellow, I., and Frey, B. (2016).
        Adversarial Autoencoders. ICLR Workshop.

    Porting:
        Mirrors ``autoenc_adv_e.R`` and replaces ``inst/python/autoenc_adv.py``.
        Fidelity: reconstruction, discriminator and generator phases, dropout,
        Gaussian prior and optimizer schedule match the legacy backend.
    """

    def __init__(self, input_size: int, encoding_size: int, **kwargs) -> None:
        AutoencBaseE.__init__(self, input_size, encoding_size)
        self._initialize(input_size=input_size, encoding_size=encoding_size, **kwargs)

    def transform(self, data, **kwargs) -> np.ndarray:
        return self._transform(data, reconstruct=False)


class AutoencAdvEd(_AdversarialAutoencoderMixin, AutoencBaseEd):
    """Adversarial encoder-decoder returning reconstructed observations.

    References:
        Makhzani, A., Shlens, J., Jaitly, N., Goodfellow, I., and Frey, B. (2016).
        Adversarial Autoencoders. ICLR Workshop.

    Porting:
        Mirrors ``autoenc_adv_ed.R`` and replaces ``inst/python/autoenc_adv.py``.
        Fidelity: reconstruction, discriminator and generator phases, dropout,
        Gaussian prior and optimizer schedule match the legacy backend.
    """

    def __init__(self, input_size: int, encoding_size: int, **kwargs) -> None:
        AutoencBaseEd.__init__(self, input_size, encoding_size)
        self._initialize(input_size=input_size, encoding_size=encoding_size, **kwargs)

    def transform(self, data, **kwargs) -> np.ndarray:
        return self._transform(data, reconstruct=True)
