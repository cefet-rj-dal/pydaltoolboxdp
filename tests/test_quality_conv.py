import json
from pathlib import Path

import numpy as np

from pydaltoolboxdp import AutoencConvE, AutoencConvEd


def test_convolutional_autoencoders_match_frozen_legacy_backend():
    reference_path = Path(__file__).parents[1] / "develop" / "parity" / "conv_ref.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    data = np.linspace(0.05, 0.95, 48, dtype=np.float32).reshape(12, 4)
    options = {
        "hidden_channels": 3,
        "bottleneck_channels": 4,
        "kernel_size": 3,
        "batch_size": 4,
        "epochs": 4,
        "learning_rate": 0.001,
        "random_state": 37,
    }

    encoder = AutoencConvE(4, 2, **options).fit(data)
    decoder = AutoencConvEd(4, 2, **options).fit(data)

    np.testing.assert_allclose(encoder.train_loss_, reference["train_loss"], atol=1e-7)
    np.testing.assert_allclose(decoder.train_loss_, reference["train_loss"], atol=1e-7)
    np.testing.assert_allclose(encoder.val_loss_, reference["val_loss"], atol=1e-7)
    np.testing.assert_allclose(encoder.transform(data), reference["encoded"], atol=1e-7)
    np.testing.assert_allclose(
        decoder.transform(data), reference["reconstructed"], atol=1e-7
    )
