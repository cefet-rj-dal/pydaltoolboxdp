import numpy as np
import pytest
from pyharbinger import Harbinger

from pydaltoolboxdp import HanAutoencoder, HcpWaypoint


@pytest.fixture
def anomaly_series():
    values = np.sin(np.linspace(0, 5 * np.pi, 96))
    values[70] += 4.0
    return values


def test_han_autoencoder_implements_harbinger_contract(anomaly_series):
    detector = HanAutoencoder(
        6,
        3,
        encoder_hidden_sizes=(8,),
        epochs=2,
        batch_size=8,
        random_state=101,
    ).fit(anomaly_series)
    detection = detector.detect(anomaly_series)

    assert isinstance(detector, Harbinger)
    assert detection.shape == (len(anomaly_series), 3)
    assert detection.columns.tolist() == ["idx", "event", "type"]
    assert detection.attrs["res"].shape == anomaly_series.shape
    assert np.all(detection.attrs["res"][:5] == 0)
    assert np.isfinite(detection.attrs["res"]).all()


def test_han_autoencoder_seed_is_reproducible(anomaly_series):
    options = {
        "encoder_hidden_sizes": (7,),
        "epochs": 2,
        "batch_size": 8,
        "random_state": 103,
    }
    first = HanAutoencoder(5, 2, **options).fit(anomaly_series).detect(anomaly_series)
    second = HanAutoencoder(5, 2, **options).fit(anomaly_series).detect(anomaly_series)

    np.testing.assert_allclose(first.attrs["res"], second.attrs["res"], atol=1e-7)
    np.testing.assert_array_equal(first["event"], second["event"])


def test_hcp_waypoint_exposes_adaptive_diagnostics():
    baseline = np.sin(np.linspace(0, 4 * np.pi, 70)) * 0.1
    changed = 1.5 + np.sin(np.linspace(0, 3 * np.pi, 60)) * 0.1
    serie = np.concatenate([baseline, changed])
    detector = HcpWaypoint(
        6,
        3,
        warmup=40,
        retrain_size=20,
        buffer_size=25,
        h_low=0.5,
        h_high=1.0,
        epochs_init=2,
        epochs_retrain=1,
        encoder_hidden_sizes=(8,),
        batch_size=8,
        random_state=107,
    ).fit(serie)
    detection = detector.detect(serie)

    assert isinstance(detector, Harbinger)
    assert detection.shape == (len(serie), 3)
    assert detector.tau_ is not None
    assert detector.tau_path_.shape == serie.shape
    assert detection.attrs["tau"] == detector.tau_
    assert detection.attrs["drift_indices"] == detector.drift_indices_
    assert all(0 <= index < len(serie) for index in detector.drift_indices_)


def test_event_detectors_restore_missing_positions(anomaly_series):
    serie = anomaly_series.copy()
    serie[11] = np.nan
    detector = HanAutoencoder(5, 2, epochs=1, batch_size=8, random_state=109)
    detection = detector.fit(serie).detect(serie)

    assert len(detection) == len(serie)
    assert not detection.loc[11, "event"]
    assert np.isnan(detection.attrs["res"][11])


@pytest.mark.parametrize("detector", [HanAutoencoder, HcpWaypoint])
def test_event_detectors_validate_sizes(detector):
    with pytest.raises(ValueError, match="positive"):
        detector(0, 2)
