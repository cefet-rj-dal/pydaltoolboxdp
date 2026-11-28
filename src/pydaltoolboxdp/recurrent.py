"""Long short-term memory Deep ++ autoencoders."""

from __future__ import annotations

import numpy as np
import torch
from pydaltoolbox import AutoencBaseE, AutoencBaseEd
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .autoencoder import _DenseAutoencoderMixin

__all__ = ["AutoencLstmE", "AutoencLstmEd"]


class _LstmEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        latent_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        effective_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.projection = nn.Linear(hidden_size, latent_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.projection(hidden[-1]).unsqueeze(1)


class _LstmDecoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        sequence_length: int,
        latent_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        effective_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        self.sequence_length = sequence_length
        self.latent_to_hidden = nn.Linear(latent_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        repeated = self.latent_to_hidden(x).repeat(1, self.sequence_length, 1)
        output, _ = self.lstm(repeated)
        return self.output_layer(output)


class _LstmNetwork(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        sequence_length: int,
        latent_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = _LstmEncoder(
            feature_dim, latent_size, hidden_size, num_layers, dropout
        )
        self.decoder = _LstmDecoder(
            feature_dim,
            sequence_length,
            latent_size,
            hidden_size,
            num_layers,
            dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class _LstmAutoencoderMixin(_DenseAutoencoderMixin):
    def _initialize(
        self,
        *,
        lstm_hidden_size: int | None = None,
        sequence_length: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        input_size = int(kwargs["input_size"])
        encoding_size = int(kwargs["encoding_size"])
        resolved_sequence_length = (
            input_size if sequence_length is None else int(sequence_length)
        )
        if resolved_sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if input_size % resolved_sequence_length != 0:
            raise ValueError("input_size must be divisible by sequence_length")
        resolved_hidden_size = (
            encoding_size if lstm_hidden_size is None else int(lstm_hidden_size)
        )
        if resolved_hidden_size <= 0:
            raise ValueError("lstm_hidden_size must be positive")
        if int(num_layers) <= 0:
            raise ValueError("num_layers must be positive")
        if not 0 <= float(dropout) < 1:
            raise ValueError("dropout must be in the interval [0, 1)")

        self.sequence_length = resolved_sequence_length
        self.feature_dim = input_size // resolved_sequence_length
        self.lstm_hidden_size = resolved_hidden_size
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        super()._initialize(**kwargs)

    def _build_model(self) -> None:
        self._seed()
        self.device_ = torch.device(self.device)
        self.model = _LstmNetwork(
            self.feature_dim,
            self.sequence_length,
            self.encoding_size,
            self.lstm_hidden_size,
            self.num_layers,
            self.dropout,
        ).float().to(self.device_)

    def _loader(self, array: np.ndarray, *, shuffle: bool) -> DataLoader:
        sequence = np.array(
            array.reshape(len(array), self.sequence_length, self.feature_dim),
            dtype=np.float32,
            copy=True,
        )
        tensor = torch.from_numpy(sequence)
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
                encoded = self.model.encoder(features.to(self.device_))
                result.append(encoded.cpu().numpy().reshape(len(features), -1))
        return np.concatenate(result, axis=0)

    def reconstruct(self, data) -> np.ndarray:
        self._require_fitted()
        loader = self._loader(self._array(data), shuffle=False)
        result = []
        self.model.eval()
        with torch.no_grad():
            for features, _ in loader:
                decoded = self.model(features.to(self.device_))
                result.append(decoded.cpu().numpy().reshape(len(features), -1))
        return np.concatenate(result, axis=0)


class AutoencLstmE(_LstmAutoencoderMixin, AutoencBaseE):
    """LSTM encoder for temporal dependencies within observation windows.

    References:
        Hochreiter, S., and Schmidhuber, J. (1997). Long Short-Term Memory.
        Neural Computation, 9(8), 1735-1780.

    Porting:
        Mirrors ``autoenc_lstm_e.R`` and replaces
        ``inst/python/autoenc_lstm.py``. Fidelity: sequence reshaping, recurrent
        encoder, latent projection, decoder, optimizer and training contract match
        the legacy backend.
    """

    def __init__(self, input_size: int, encoding_size: int, **kwargs) -> None:
        AutoencBaseE.__init__(self, input_size, encoding_size)
        self._initialize(input_size=input_size, encoding_size=encoding_size, **kwargs)

    def transform(self, data, **kwargs) -> np.ndarray:
        return self.encode(data)


class AutoencLstmEd(_LstmAutoencoderMixin, AutoencBaseEd):
    """LSTM encoder-decoder that reconstructs observation windows.

    References:
        Hochreiter, S., and Schmidhuber, J. (1997). Long Short-Term Memory.
        Neural Computation, 9(8), 1735-1780.

    Porting:
        Mirrors ``autoenc_lstm_ed.R`` and replaces
        ``inst/python/autoenc_lstm.py``. Fidelity: sequence reshaping, recurrent
        encoder, latent projection, decoder, optimizer and training contract match
        the legacy backend.
    """

    def __init__(self, input_size: int, encoding_size: int, **kwargs) -> None:
        AutoencBaseEd.__init__(self, input_size, encoding_size)
        self._initialize(input_size=input_size, encoding_size=encoding_size, **kwargs)

    def transform(self, data, **kwargs) -> np.ndarray:
        return self.reconstruct(data)
