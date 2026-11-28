"""One-dimensional convolutional Deep ++ autoencoders."""

from __future__ import annotations

import numpy as np
import torch
from pydaltoolbox import AutoencBaseE, AutoencBaseEd
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .autoencoder import _DenseAutoencoderMixin

__all__ = ["AutoencConvE", "AutoencConvEd"]


class _ConvNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        encoding_size: int,
        hidden_channels: int,
        bottleneck_channels: int,
        kernel_size: int,
        negative_slope: float,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.input_size = int(input_size)
        self.encoder_features = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv1d(
                hidden_channels,
                bottleneck_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.encoder_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(bottleneck_channels * self.input_size, encoding_size),
        )
        self.decoder_projection = nn.Sequential(
            nn.Linear(encoding_size, bottleneck_channels * self.input_size),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Unflatten(1, (bottleneck_channels, self.input_size)),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                bottleneck_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.ConvTranspose1d(
                hidden_channels,
                1,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_projection(self.encoder_features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        return self.decoder(self.decoder_projection(latent))


class _ConvAutoencoderMixin(_DenseAutoencoderMixin):
    def _initialize(
        self,
        *,
        hidden_channels: int = 16,
        bottleneck_channels: int = 32,
        kernel_size: int = 3,
        **kwargs,
    ) -> None:
        if int(hidden_channels) <= 0 or int(bottleneck_channels) <= 0:
            raise ValueError("convolutional channel counts must be positive")
        if int(kernel_size) <= 0 or int(kernel_size) % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number")
        self.hidden_channels = int(hidden_channels)
        self.bottleneck_channels = int(bottleneck_channels)
        self.kernel_size = int(kernel_size)
        super()._initialize(**kwargs)

    def _build_model(self) -> None:
        self._seed()
        self.device_ = torch.device(self.device)
        self.model = _ConvNetwork(
            self.input_size,
            self.encoding_size,
            self.hidden_channels,
            self.bottleneck_channels,
            self.kernel_size,
            self.negative_slope,
        ).float().to(self.device_)

    def _loader(self, array: np.ndarray, *, shuffle: bool) -> DataLoader:
        tensor = torch.from_numpy(
            np.array(array[:, np.newaxis, :], dtype=np.float32, copy=True)
        )
        return DataLoader(
            TensorDataset(tensor, tensor),
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    def encode(self, data) -> np.ndarray:
        self._require_fitted()
        loader = self._loader(self._array(data), shuffle=False)
        result = []
        self.model.eval()
        with torch.no_grad():
            for features, _ in loader:
                encoded = self.model.encode(features.to(self.device_))
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
                result.append(decoded.cpu().numpy().squeeze(1))
        return np.concatenate(result, axis=0)


class AutoencConvE(_ConvAutoencoderMixin, AutoencBaseE):
    """Conv1D encoder for local patterns within observation windows.

    References:
        Masci, J., Meier, U., Ciresan, D., and Schmidhuber, J. (2011).
        Stacked Convolutional Auto-Encoders for Hierarchical Feature Extraction.
        ICANN 2011, 52-59.

    Porting:
        Mirrors ``autoenc_conv_e.R`` and replaces
        ``inst/python/autoenc_conv.py``. Fidelity: the one-channel reshape,
        convolutional architecture, LeakyReLU activations, linear bottleneck,
        sigmoid reconstruction, optimizer and training contract match the legacy
        backend.
    """

    def __init__(self, input_size: int, encoding_size: int, **kwargs) -> None:
        AutoencBaseE.__init__(self, input_size, encoding_size)
        self._initialize(input_size=input_size, encoding_size=encoding_size, **kwargs)

    def transform(self, data, **kwargs) -> np.ndarray:
        return self.encode(data)


class AutoencConvEd(_ConvAutoencoderMixin, AutoencBaseEd):
    """Conv1D encoder-decoder that reconstructs observation windows.

    References:
        Masci, J., Meier, U., Ciresan, D., and Schmidhuber, J. (2011).
        Stacked Convolutional Auto-Encoders for Hierarchical Feature Extraction.
        ICANN 2011, 52-59.

    Porting:
        Mirrors ``autoenc_conv_ed.R`` and replaces
        ``inst/python/autoenc_conv.py``. Fidelity: the one-channel reshape,
        convolutional architecture, LeakyReLU activations, linear bottleneck,
        sigmoid reconstruction, optimizer and training contract match the legacy
        backend.
    """

    def __init__(self, input_size: int, encoding_size: int, **kwargs) -> None:
        AutoencBaseEd.__init__(self, input_size, encoding_size)
        self._initialize(input_size=input_size, encoding_size=encoding_size, **kwargs)

    def transform(self, data, **kwargs) -> np.ndarray:
        return self.reconstruct(data)
