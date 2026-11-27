"""Dense Deep ++ autoencoders.

This module adapts the PyTorch implementation formerly embedded in
``daltoolboxdp/inst/python`` into public classes that follow the
``pydaltoolbox`` transformation contract.
"""

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

__all__ = ["AutoencE", "AutoencEd"]


class _DenseNetwork(nn.Module):
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


class _DenseAutoencoderMixin:
    def _initialize(
        self,
        *,
        input_size: int,
        encoding_size: int,
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
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if not 0 < float(val_ratio) < 1:
            raise ValueError("val_ratio must be between 0 and 1")
        validation_strategy, stopping_rule = validate_training_options(
            validation_strategy, stopping_rule
        )

        self.input_size = int(input_size)
        self.encoding_size = int(encoding_size)
        self.encoder_hidden_sizes = int_list(encoder_hidden_sizes, default=[64])
        self.decoder_hidden_sizes = (
            list(reversed(self.encoder_hidden_sizes))
            if decoder_hidden_sizes is None
            else int_list(decoder_hidden_sizes, allow_empty=True)
        )
        self.activation = str(activation)
        self.output_activation = str(output_activation)
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
        self.model = _DenseNetwork(
            self.input_size,
            self.encoding_size,
            self.encoder_hidden_sizes,
            self.decoder_hidden_sizes,
            self.activation,
            self.output_activation,
            self.negative_slope,
        ).float().to(self.device_)

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

    def _run_epoch(self, loader: DataLoader, optimizer=None) -> float:
        criterion = nn.MSELoss()
        losses: list[float] = []
        if optimizer is None:
            self.model.eval()
            with torch.no_grad():
                for features, targets in loader:
                    features = features.to(self.device_)
                    targets = targets.to(self.device_)
                    inputs = self._training_inputs(features)
                    losses.append(float(criterion(self.model(inputs), targets).item()))
        else:
            self.model.train()
            for features, targets in loader:
                features = features.to(self.device_)
                targets = targets.to(self.device_)
                inputs = self._training_inputs(features)
                optimizer.zero_grad()
                loss = criterion(self.model(inputs), targets)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else 0.0

    def _training_inputs(self, features: torch.Tensor) -> torch.Tensor:
        """Return the inputs used to optimize against the clean targets."""
        return features

    def fit(self, data, **kwargs):
        """Train the autoencoder and return ``self``.

        Training parameters may be overridden for this call. This is needed by adaptive
        consumers such as Waypoint, which retrain an existing model for fewer epochs.
        """
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

        array = self._array(data)
        self._build_model()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.learning_rate))
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
        stopper = StopController(config)
        self.train_loss_ = []
        self.val_loss_ = []
        self.epochs_done_ = 0

        train_loader = None
        validation_loader = None
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
    def train_loss(self) -> list[float]:
        """Compatibility view used by the R wrapper."""
        return self.train_loss_

    @property
    def val_loss(self) -> list[float]:
        """Compatibility view used by the R wrapper."""
        return self.val_loss_

    def _require_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError("the autoencoder must be fitted before transform")

    def encode(self, data) -> np.ndarray:
        self._require_fitted()
        loader = self._loader(self._array(data), shuffle=False)
        result = []
        self.model.eval()
        with torch.no_grad():
            for features, _ in loader:
                encoded = self.model.encoder(features.to(self.device_))
                result.append(encoded.cpu().numpy())
        return np.concatenate(result, axis=0)

    def reconstruct(self, data) -> np.ndarray:
        self._require_fitted()
        loader = self._loader(self._array(data), shuffle=False)
        result = []
        self.model.eval()
        with torch.no_grad():
            for features, _ in loader:
                decoded = self.model(features.to(self.device_))
                result.append(decoded.cpu().numpy())
        return np.concatenate(result, axis=0)


class AutoencE(_DenseAutoencoderMixin, AutoencBaseE):
    """Dense encoder that returns a learned latent representation.

    References:
        Hinton, G. E., and Salakhutdinov, R. R. (2006). Reducing the
        Dimensionality of Data with Neural Networks. Science, 313(5786):504-507.
        Paszke, A., et al. (2019). PyTorch: An Imperative Style,
        High-Performance Deep Learning Library. NeurIPS 32.

    Porting:
        Mirrors ``autoenc_e.R`` and replaces its embedded ``inst/python/autoenc.py``
        backend. Fidelity: exact in architecture and training contract; an optional
        ``random_state`` makes stochastic execution reproducible.
    """

    def __init__(self, input_size: int, encoding_size: int, **kwargs) -> None:
        AutoencBaseE.__init__(self, input_size, encoding_size)
        self._initialize(input_size=input_size, encoding_size=encoding_size, **kwargs)

    def transform(self, data, **kwargs) -> np.ndarray:
        return self.encode(data)


class AutoencEd(_DenseAutoencoderMixin, AutoencBaseEd):
    """Dense encoder-decoder that returns reconstructed observations.

    References:
        Hinton, G. E., and Salakhutdinov, R. R. (2006). Reducing the
        Dimensionality of Data with Neural Networks. Science, 313(5786):504-507.
        Paszke, A., et al. (2019). PyTorch: An Imperative Style,
        High-Performance Deep Learning Library. NeurIPS 32.

    Porting:
        Mirrors ``autoenc_ed.R`` and replaces its embedded ``inst/python/autoenc.py``
        backend. Fidelity: exact in architecture and training contract; an optional
        ``random_state`` makes stochastic execution reproducible.
    """

    def __init__(self, input_size: int, encoding_size: int, **kwargs) -> None:
        AutoencBaseEd.__init__(self, input_size, encoding_size)
        self._initialize(input_size=input_size, encoding_size=encoding_size, **kwargs)

    def transform(self, data, **kwargs) -> np.ndarray:
        return self.reconstruct(data)


class _DenoisingAutoencoderMixin(_DenseAutoencoderMixin):
    def _initialize(self, *, noise_factor: float = 0.3, **kwargs) -> None:
        if float(noise_factor) < 0:
            raise ValueError("noise_factor must be non-negative")
        self.noise_factor = float(noise_factor)
        super()._initialize(**kwargs)

    def _training_inputs(self, features: torch.Tensor) -> torch.Tensor:
        return features + torch.randn_like(features) * self.noise_factor


class AutoencDenoiseE(_DenoisingAutoencoderMixin, AutoencBaseE):
    """Denoising encoder that learns latent representations from corrupted inputs.

    References:
        Vincent, P., Larochelle, H., Bengio, Y., and Manzagol, P. A. (2008).
        Extracting and Composing Robust Features with Denoising Autoencoders.
        Proceedings of ICML 2008, 1096-1103.

    Porting:
        Mirrors ``autoenc_denoise_e.R`` and replaces
        ``inst/python/autoenc_denoise.py``. Fidelity: architecture, Gaussian input
        corruption and clean reconstruction targets match the legacy backend.
    """

    def __init__(self, input_size: int, encoding_size: int, **kwargs) -> None:
        AutoencBaseE.__init__(self, input_size, encoding_size)
        self._initialize(input_size=input_size, encoding_size=encoding_size, **kwargs)

    def transform(self, data, **kwargs) -> np.ndarray:
        return self.encode(data)


class AutoencDenoiseEd(_DenoisingAutoencoderMixin, AutoencBaseEd):
    """Denoising encoder-decoder that reconstructs clean observations.

    References:
        Vincent, P., Larochelle, H., Bengio, Y., and Manzagol, P. A. (2008).
        Extracting and Composing Robust Features with Denoising Autoencoders.
        Proceedings of ICML 2008, 1096-1103.

    Porting:
        Mirrors ``autoenc_denoise_ed.R`` and replaces
        ``inst/python/autoenc_denoise.py``. Fidelity: architecture, Gaussian input
        corruption and clean reconstruction targets match the legacy backend.
    """

    def __init__(self, input_size: int, encoding_size: int, **kwargs) -> None:
        AutoencBaseEd.__init__(self, input_size, encoding_size)
        self._initialize(input_size=input_size, encoding_size=encoding_size, **kwargs)

    def transform(self, data, **kwargs) -> np.ndarray:
        return self.reconstruct(data)
