import json
from pathlib import Path

import numpy as np
import torch

from pydaltoolboxdp import AutoencVariationalE, AutoencVariationalEd


def test_variational_autoencoders_match_frozen_legacy_backend():
    reference_path = (
        Path(__file__).parents[1] / "develop" / "parity" / "variational_ref.json"
    )
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    data = np.linspace(0.1, 0.9, 48, dtype=np.float32).reshape(12, 4)
    options = {
        "encoder_hidden_sizes": [5],
        "decoder_hidden_sizes": [5],
        "activation": "leaky_relu",
        "negative_slope": 0.01,
        "output_activation": "sigmoid",
        "reconstruction_loss": "bce",
        "batch_size": 4,
        "epochs": 3,
        "learning_rate": 0.001,
        "random_state": 101,
    }

    encoder = AutoencVariationalE(4, 2, **options).fit(data)
    decoder = AutoencVariationalEd(4, 2, **options).fit(data)
    torch.manual_seed(107)
    encoded = encoder.transform(data)
    torch.manual_seed(109)
    reconstructed = decoder.transform(data)

    np.testing.assert_allclose(encoder.train_loss_, reference["train_loss"], atol=1e-7)
    np.testing.assert_allclose(decoder.train_loss_, reference["train_loss"], atol=1e-7)
    np.testing.assert_allclose(encoder.val_loss_, reference["val_loss"], atol=1e-7)
    np.testing.assert_allclose(encoded, reference["encoded"], atol=1e-7)
    np.testing.assert_allclose(reconstructed, reference["reconstructed"], atol=1e-7)
