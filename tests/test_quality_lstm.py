import json
from pathlib import Path

import numpy as np

from pydaltoolboxdp import AutoencLstmE, AutoencLstmEd


def test_lstm_autoencoders_match_frozen_legacy_backend():
    reference_path = Path(__file__).parents[1] / "develop" / "parity" / "lstm_ref.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    data = np.linspace(-0.8, 0.8, 72, dtype=np.float32).reshape(12, 6)
    options = {
        "lstm_hidden_size": 4,
        "sequence_length": 3,
        "num_layers": 1,
        "dropout": 0.0,
        "batch_size": 4,
        "epochs": 3,
        "learning_rate": 0.001,
        "random_state": 47,
    }

    encoder = AutoencLstmE(6, 2, **options).fit(data)
    decoder = AutoencLstmEd(6, 2, **options).fit(data)

    np.testing.assert_allclose(encoder.train_loss_, reference["train_loss"], atol=1e-7)
    np.testing.assert_allclose(decoder.train_loss_, reference["train_loss"], atol=1e-7)
    np.testing.assert_allclose(encoder.val_loss_, reference["val_loss"], atol=1e-7)
    np.testing.assert_allclose(encoder.transform(data), reference["encoded"], atol=1e-7)
    np.testing.assert_allclose(
        decoder.transform(data), reference["reconstructed"], atol=1e-7
    )
