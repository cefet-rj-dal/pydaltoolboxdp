import numpy as np
import pandas as pd
import pytest
from pydaltoolbox import Classification, Regression

from pydaltoolboxdp import SkclaGb, TorchClaMlp, TorchRegMlp


@pytest.fixture
def classification_data():
    rng = np.random.default_rng(17)
    features = rng.normal(size=(30, 3)).astype(np.float32)
    labels = np.where(features[:, 0] + 0.5 * features[:, 1] > 0, "yes", "no")
    return pd.DataFrame({"x1": features[:, 0], "x2": features[:, 1],
                         "x3": features[:, 2], "class": labels})


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(23)
    features = rng.normal(size=(30, 3)).astype(np.float32)
    target = 1.5 * features[:, 0] - 0.75 * features[:, 1] + 0.25
    return pd.DataFrame({"x1": features[:, 0], "x2": features[:, 1],
                         "x3": features[:, 2], "target": target})


def test_skcla_gb_implements_contract_and_orders_probabilities(classification_data):
    model = SkclaGb(
        "class", ["yes", "no"], n_estimators=8, max_depth=2, random_state=5
    ).fit(classification_data)
    prediction = model.predict(classification_data)

    assert isinstance(model, Classification)
    assert list(prediction.columns) == ["yes", "no"]
    np.testing.assert_allclose(prediction.sum(axis=1), 1.0, atol=1e-7)


def test_torch_classifier_is_seeded_and_returns_scores(classification_data):
    options = {
        "input_size": 3, "hidden_sizes": [5], "dropout": 0.0,
        "epochs": 3, "batch_size": 6, "random_state": 29,
    }
    first = TorchClaMlp("class", ["yes", "no"], **options).fit(classification_data)
    second = TorchClaMlp("class", ["yes", "no"], **options).fit(classification_data)

    assert isinstance(first, Classification)
    assert first.epochs_done == 3
    np.testing.assert_allclose(
        first.predict(classification_data), second.predict(classification_data), atol=1e-7
    )
    np.testing.assert_allclose(first.train_loss_hist, second.train_loss_hist, atol=1e-7)


def test_torch_regressor_infers_input_size_and_is_seeded(regression_data):
    options = {
        "hidden_sizes": [5], "dropout": 0.0, "epochs": 3,
        "batch_size": 6, "random_state": 31,
    }
    first = TorchRegMlp("target", **options).fit(regression_data)
    second = TorchRegMlp("target", **options).fit(regression_data)

    assert isinstance(first, Regression)
    assert first.input_size == 3
    assert first.predict(regression_data).shape == (30,)
    np.testing.assert_allclose(
        first.predict(regression_data), second.predict(regression_data), atol=1e-7
    )
    np.testing.assert_allclose(first.train_loss_hist, second.train_loss_hist, atol=1e-7)


def test_torch_classifier_checks_feature_count(classification_data):
    with pytest.raises(ValueError, match="fitted data has 3 predictors"):
        TorchClaMlp(
            "class", ["yes", "no"], input_size=4, hidden_sizes=[5], epochs=1
        ).fit(classification_data)


def test_torch_regressor_validation_and_stopping(regression_data):
    model = TorchRegMlp(
        "target", hidden_sizes=[4], dropout=0.0, epochs=8, batch_size=4,
        stopping_rule="patience", patience=1, random_state=37,
    ).fit(regression_data)

    assert model.val_loss_hist
    assert model.epochs_done <= 8
