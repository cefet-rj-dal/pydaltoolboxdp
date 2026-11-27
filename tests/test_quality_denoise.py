import json
from pathlib import Path

import numpy as np

from pydaltoolboxdp import AutoencDenoiseE, AutoencDenoiseEd


def test_denoising_autoencoders_match_frozen_legacy_backend():
    reference_path = (
        Path(__file__).parents[1] / "develop" / "parity" / "denoise_ref.json"
    )
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    data = np.random.default_rng(71).normal(size=(12, 4)).astype(np.float32)
    options = {
        "noise_factor": 0.2,
        "encoder_hidden_sizes": [5],
        "decoder_hidden_sizes": [5],
        "batch_size": 4,
        "epochs": 3,
        "learning_rate": 0.001,
        "random_state": 23,
    }

    encoder = AutoencDenoiseE(4, 2, **options).fit(data)
    decoder = AutoencDenoiseEd(4, 2, **options).fit(data)

    np.testing.assert_allclose(encoder.train_loss_, reference["train_loss"], atol=1e-7)
    np.testing.assert_allclose(decoder.train_loss_, reference["train_loss"], atol=1e-7)
    np.testing.assert_allclose(encoder.transform(data), reference["encoded"], atol=1e-7)
    np.testing.assert_allclose(
        decoder.transform(data), reference["reconstructed"], atol=1e-7
    )
