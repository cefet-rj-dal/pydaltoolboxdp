import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pydaltoolboxdp import SkclaGb, TorchClaMlp, TorchRegMlp

REFERENCE = json.loads(
    (Path(__file__).parents[1] / "develop/parity/supervised_ref.json").read_text()
)


def _data():
    classification = pd.DataFrame({
        "x1": [-1.2, -0.8, -0.3, 0.1, 0.4, 0.9, 1.2, 1.5],
        "x2": [0.2, -0.1, 0.5, -0.4, 0.7, 0.3, -0.2, 0.8],
        "class": ["no", "no", "no", "yes", "yes", "yes", "yes", "yes"],
    })
    regression = classification[["x1", "x2"]].copy()
    regression["target"] = [
        float(1.5 * x - 0.4 * y + 0.2)
        for x, y in regression[["x1", "x2"]].to_numpy()
    ]
    return classification, regression


def test_gradient_boosting_matches_frozen_legacy_probabilities():
    classification, _ = _data()
    np.random.seed(REFERENCE["seed_gb"])
    model = SkclaGb(
        "class", ["yes", "no"], n_estimators=7, learning_rate=0.08,
        max_depth=2, subsample=0.75,
    ).fit(classification)
    np.testing.assert_allclose(
        model.predict(classification), REFERENCE["gb_scores"], atol=1e-7
    )


def test_torch_classifier_matches_frozen_legacy_loss_and_probabilities():
    classification, _ = _data()
    np.random.seed(REFERENCE["seed_classifier"])
    torch.manual_seed(REFERENCE["seed_classifier"])
    model = TorchClaMlp(
        "class", ["yes", "no"], 2, [4], dropout=0.0,
        epochs=4, lr=0.01, batch_size=4,
    ).fit(classification)
    np.testing.assert_allclose(model.train_loss_, REFERENCE["classifier_loss"], atol=1e-7)
    np.testing.assert_allclose(
        model.predict(classification), REFERENCE["classifier_scores"], atol=1e-7
    )


def test_torch_regressor_matches_frozen_legacy_loss_and_prediction():
    _, regression = _data()
    np.random.seed(REFERENCE["seed_regressor"])
    torch.manual_seed(REFERENCE["seed_regressor"])
    model = TorchRegMlp(
        "target", [4], input_size=2, dropout=0.0,
        epochs=4, lr=0.01, batch_size=4,
    ).fit(regression)
    np.testing.assert_allclose(model.train_loss_, REFERENCE["regressor_loss"], atol=1e-7)
    np.testing.assert_allclose(
        model.predict(regression), REFERENCE["regressor_prediction"], atol=1e-7
    )
