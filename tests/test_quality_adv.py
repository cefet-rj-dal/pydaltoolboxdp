import json
from pathlib import Path

import numpy as np

from pydaltoolboxdp import AutoencAdvE, AutoencAdvEd


def test_adversarial_autoencoders_match_frozen_legacy_backend():
    reference_path = Path(__file__).parents[1] / "develop" / "parity" / "adv_ref.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    data = np.linspace(0.1, 0.9, 48, dtype=np.float32).reshape(12, 4)
    options = {
        "encoder_hidden_sizes": [5],
        "decoder_hidden_sizes": [5],
        "discriminator_hidden_sizes": [5],
        "activation": "relu",
        "dropout": 0.1,
        "latent_prior_scale": 1.5,
        "batch_size": 4,
        "epochs": 2,
        "learning_rate": 0.001,
        "random_state": 113,
    }

    encoder = AutoencAdvE(4, 2, **options).fit(data)
    decoder = AutoencAdvEd(4, 2, **options).fit(data)

    np.testing.assert_allclose(encoder.train_loss_, reference["train_loss"], atol=1e-7)
    np.testing.assert_allclose(decoder.train_loss_, reference["train_loss"], atol=1e-7)
    np.testing.assert_allclose(encoder.val_loss_, reference["val_loss"], atol=1e-7)
    np.testing.assert_allclose(encoder.transform(data), reference["encoded"], atol=1e-7)
    np.testing.assert_allclose(
        decoder.transform(data), reference["reconstructed"], atol=1e-7
    )
