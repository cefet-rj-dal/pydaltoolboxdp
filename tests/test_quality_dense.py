import json
from pathlib import Path

import numpy as np

from pydaltoolboxdp import AutoencE, AutoencEd


def test_dense_autoencoders_match_frozen_legacy_backend():
    reference_path = (
        Path(__file__).parents[1] / "develop" / "parity" / "dense_ref.json"
    )
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    data = np.linspace(-1.5, 1.5, 48, dtype=np.float32).reshape(12, 4)
    options = {
        "encoder_hidden_sizes": [6],
        "decoder_hidden_sizes": [6],
        "activation": "relu",
        "output_activation": "none",
        "negative_slope": 0.01,
        "validation_strategy": "static",
        "stopping_rule": "none",
        "batch_size": 4,
        "epochs": 4,
        "learning_rate": 0.001,
        "random_state": 23,
    }

    encoder = AutoencE(4, 2, **options).fit(data)
    decoder = AutoencEd(4, 2, **options).fit(data)

    np.testing.assert_allclose(encoder.train_loss_, reference["train_loss"], atol=1e-7)
    np.testing.assert_allclose(decoder.train_loss_, reference["train_loss"], atol=1e-7)
    np.testing.assert_allclose(encoder.val_loss_, reference["val_loss"], atol=1e-7)
    np.testing.assert_allclose(encoder.transform(data), reference["encoded"], atol=1e-7)
    np.testing.assert_allclose(
        decoder.transform(data), reference["reconstructed"], atol=1e-7
    )
