import json
from pathlib import Path

import numpy as np

from pydaltoolboxdp import HanAutoencoder, HcpWaypoint

REFERENCE = json.loads(
    (Path(__file__).parents[1] / "develop" / "parity" / "events_ref.json").read_text()
)


def _series():
    values = np.sin(np.linspace(0, 8 * np.pi, 130)) * 0.1
    values[70:] += 1.5
    values[103] += 2.0
    return values


def test_han_autoencoder_frozen_quality_reference():
    detector = HanAutoencoder(
        6,
        3,
        encoder_hidden_sizes=(8,),
        epochs=2,
        batch_size=8,
        random_state=307,
    ).fit(_series())
    detection = detector.detect(_series())

    assert np.flatnonzero(detection["event"]).tolist() == REFERENCE["han"]["events"]
    np.testing.assert_allclose(
        detection.attrs["res"][[5, 40, 70, 103, 129]],
        REFERENCE["han"]["residuals"],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        detection.attrs["threshold"], REFERENCE["han"]["threshold"], atol=1e-7
    )


def test_hcp_waypoint_frozen_quality_reference():
    detector = HcpWaypoint(
        6,
        3,
        warmup=40,
        retrain_size=20,
        buffer_size=25,
        k_factor=0.35,
        h_low=0.5,
        h_high=1.0,
        epochs_init=2,
        epochs_retrain=2,
        encoder_hidden_sizes=(8,),
        epochs=2,
        batch_size=8,
        random_state=311,
    ).fit(_series())
    detection = detector.detect(_series())

    assert np.flatnonzero(detection["event"]).tolist() == REFERENCE["waypoint"]["events"]
    assert detector.drift_indices_ == REFERENCE["waypoint"]["drift_indices"]
    np.testing.assert_allclose(
        detection.attrs["res"][[40, 70, 102, 103, 129]],
        REFERENCE["waypoint"]["residuals"],
        atol=1e-7,
    )
    np.testing.assert_allclose(detection.attrs["tau"], REFERENCE["waypoint"]["tau"], atol=1e-7)
