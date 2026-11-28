"""Deep ++ event detectors built on the pyharbinger contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyharbinger import Harbinger
from pyharbinger.harbinger import as_series_array
from pytspredit import TsNormGminmax, ts_data

from .autoencoder import AutoencEd

__all__ = ["HanAutoencoder", "HcpWaypoint"]


def _finite_series(serie) -> np.ndarray:
    if serie is None:
        raise ValueError("No data was provided for computation")
    values = as_series_array(serie)
    return values[np.isfinite(values)]


def _reconstruct(model, windows) -> np.ndarray:
    reconstructed = model.transform(windows)
    return np.asarray(reconstructed, dtype=float)


class HanAutoencoder(Harbinger):
    """Reconstruction-based anomaly detector over sliding windows.

    The detector normalizes overlapping windows, trains an encoder-decoder on
    them, and applies the Harbinger residual filtering pipeline to the summed
    reconstruction error of each window.

    References:
        Sakurada, M., and Yairi, T. (2014). Anomaly Detection Using
        Autoencoders with Nonlinear Dimensionality Reduction. MLSDA 2014.
        Salles, R., et al. (2020). Harbinger: Um framework para integração e
        análise de métodos de detecção de eventos em séries temporais. SBBD.

    Porting:
        Mirrors ``han_autoencoder.R``. Fidelity: exact orchestration, global
        min-max preprocessing, residual aggregation, thresholding and index
        restoration; the concrete autoencoder is supplied by Deep ++.
    """

    def __init__(
        self,
        input_size: int,
        encoding_size: int,
        encoder_class=AutoencEd,
        *,
        encoder=None,
        preprocess=None,
        **encoder_kwargs,
    ) -> None:
        super().__init__()
        if int(input_size) <= 0 or int(encoding_size) <= 0:
            raise ValueError("input_size and encoding_size must be positive")
        if encoder is not None and encoder_kwargs:
            raise ValueError("encoder_kwargs cannot be used with an encoder instance")
        self.input_size = int(input_size)
        self.encoding_size = int(encoding_size)
        self.model = (
            encoder
            if encoder is not None
            else encoder_class(self.input_size, self.encoding_size, **encoder_kwargs)
        )
        self.preprocess = preprocess if preprocess is not None else TsNormGminmax()

    def fit(self, serie, **kwargs) -> HanAutoencoder:
        windows = ts_data(_finite_series(serie), self.input_size)
        self.preprocess.fit(windows)
        normalized = self.preprocess.transform(windows)
        self.model.fit(normalized, **kwargs)
        return self

    def detect(self, serie, **kwargs) -> pd.DataFrame:
        if serie is None:
            raise ValueError("No data was provided for computation")
        self._store_refs(serie)
        windows = ts_data(self.serie, self.input_size)
        normalized = self.preprocess.transform(windows)
        values = windows.iloc[:, -1].to_numpy(dtype=float)
        reconstructed = _reconstruct(self.model, normalized)
        residuals = self.har_distance(
            np.sum(normalized.to_numpy(dtype=float) - reconstructed, axis=1)
        )
        outliers = self.har_outliers(residuals)
        selection = self.har_outliers_check(outliers, residuals, values)

        padding = self.input_size - 1
        full_residuals = np.concatenate([np.zeros(padding), residuals])
        anomalies = np.concatenate(
            [np.zeros(padding, dtype=bool), selection.mask]
        )
        return self._restore_refs(
            anomalies=anomalies,
            res=full_residuals,
            threshold=selection.threshold,
        )


class HcpWaypoint(Harbinger):
    """Adaptive autoencoder and bilateral-CUSUM change-point detector.

    An autoencoder learns a reference regime from normalized sliding windows.
    Reconstruction errors are standardized against a recent buffer and fed to
    bilateral CUSUM supervisors; confirmed drift retrains the representation on
    the latest regime.

    References:
        Ygorra, B., et al. (2021). Monitoring loss of tropical forest cover
        from Sentinel-1 time-series: A CuSum-based approach.
        De Ryck, T., et al. (2021). Change Point Detection in Time Series Data
        Using Autoencoders with a Time-Invariant Representation.
        Corizzo, R., et al. (2022). CPDGA: Change point driven growing
        auto-encoder for lifelong anomaly detection.

    Porting:
        Mirrors ``hcp_waypoint.R``. Fidelity: exact warm-up, rolling residual
        standardization, bilateral CUSUM, drift confirmation and adaptive
        retraining; indices exposed by Python remain zero-based.
    """

    def __init__(
        self,
        input_size: int,
        encoding_size: int,
        warmup: int = 500,
        retrain_size: int = 300,
        buffer_size: int = 100,
        k_factor: float = 0.35,
        h_low: float = 3.5,
        h_high: float = 6.0,
        prob_tau: float = 0.997,
        epochs_init: int = 40,
        epochs_retrain: int = 20,
        encoder_class=AutoencEd,
        *,
        encoder=None,
        preprocess=None,
        **encoder_kwargs,
    ) -> None:
        super().__init__()
        if int(input_size) <= 0 or int(encoding_size) <= 0:
            raise ValueError("input_size and encoding_size must be positive")
        if min(int(warmup), int(retrain_size), int(buffer_size)) <= 0:
            raise ValueError("warmup, retrain_size and buffer_size must be positive")
        if not 0 < float(prob_tau) < 1:
            raise ValueError("prob_tau must be between 0 and 1")
        if encoder is not None and encoder_kwargs:
            raise ValueError("encoder_kwargs cannot be used with an encoder instance")
        self.input_size = int(input_size)
        self.encoding_size = int(encoding_size)
        self.warmup = int(warmup)
        self.retrain_size = int(retrain_size)
        self.buffer_size = int(buffer_size)
        self.k_factor = float(k_factor)
        self.h_low = float(h_low)
        self.h_high = float(h_high)
        self.prob_tau = float(prob_tau)
        self.epochs_init = int(epochs_init)
        self.epochs_retrain = int(epochs_retrain)
        self.model = (
            encoder
            if encoder is not None
            else encoder_class(self.input_size, self.encoding_size, **encoder_kwargs)
        )
        self.preprocess = preprocess if preprocess is not None else TsNormGminmax()
        self.tau_: float | None = None
        self.tau_path_: np.ndarray | None = None
        self.drift_indices_: list[int] = []
        self._fitted = False

    def fit(self, serie, **kwargs) -> HcpWaypoint:
        values = _finite_series(serie)
        training = values[: self.warmup]
        windows = ts_data(training, self.input_size)
        self.preprocess.fit(windows)
        normalized = self.preprocess.transform(windows)
        self.model.fit(normalized, epochs=self.epochs_init, **kwargs)
        self._fitted = True
        return self

    @staticmethod
    def _mse(model, normalized) -> np.ndarray:
        matrix = normalized.to_numpy(dtype=float)
        reconstructed = _reconstruct(model, normalized)
        return np.mean((matrix - reconstructed) ** 2, axis=1)

    def detect(self, serie, **kwargs) -> pd.DataFrame:
        if serie is None:
            raise ValueError("No data was provided for computation")
        self._store_refs(serie)
        if not self._fitted:
            self.fit(self.serie, **kwargs)

        n = len(self.serie)
        residuals = np.full(n, np.nan)
        tau_path = np.full(n, np.nan)
        changes = np.zeros(n, dtype=bool)
        drifts: list[int] = []
        positive = 0.0
        negative = 0.0
        initial_end = min(self.warmup, n)
        last_retrain_position = initial_end

        training = self.serie[:initial_end]
        training_windows = ts_data(training, self.input_size)
        training_normalized = self.preprocess.transform(training_windows)
        training_mse = self._mse(self.model, training_normalized)
        tau = float(np.quantile(training_mse, self.prob_tau))

        start_index = max(initial_end, self.input_size)
        for index in range(start_index, n):
            position = index + 1
            window = self.serie[index - self.input_size + 1 : position]
            window_frame = pd.DataFrame([window], columns=training_windows.columns)
            normalized = self.preprocess.transform(window_frame)
            mse = float(self._mse(self.model, normalized)[0])
            residuals[index] = mse
            tau_path[index] = tau

            if position <= last_retrain_position + 50:
                continue
            buffer_start = max(0, position - self.buffer_size - 1)
            buffer = residuals[buffer_start:index]
            buffer = buffer[np.isfinite(buffer)]
            if len(buffer) <= 20 or np.std(buffer, ddof=1) <= 0:
                continue

            z_score = (mse - np.mean(buffer)) / (np.std(buffer, ddof=1) + 1e-5)
            positive = max(0.0, positive + z_score - self.k_factor)
            negative = min(0.0, negative + z_score + self.k_factor)
            warning = positive > self.h_low or negative < -self.h_low
            drift = positive > self.h_high or negative < -self.h_high
            if warning:
                tau_path[index] = tau

            if not drift or position - last_retrain_position <= self.retrain_size:
                continue
            changes[index] = True
            drifts.append(index)
            retrain_start = max(0, position - self.retrain_size)
            new_windows = ts_data(self.serie[retrain_start:position], self.input_size)
            self.preprocess = TsNormGminmax().fit(new_windows)
            new_normalized = self.preprocess.transform(new_windows)
            self.model.fit(new_normalized, epochs=self.epochs_retrain, **kwargs)
            tau = float(np.quantile(self._mse(self.model, new_normalized), self.prob_tau))
            positive = 0.0
            negative = 0.0
            last_retrain_position = position

        self.tau_ = tau
        self.tau_path_ = tau_path
        self.drift_indices_ = drifts
        detection = self._restore_refs(change_points=changes, res=residuals, threshold=tau)
        detection.attrs["tau"] = tau
        detection.attrs["drift_indices"] = drifts
        return detection
