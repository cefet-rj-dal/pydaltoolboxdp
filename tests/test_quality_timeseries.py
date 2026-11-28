import json
from pathlib import Path

import numpy as np
import pandas as pd

from pydaltoolboxdp import TorchTsMlp, TsConv1D, TsLstm

REFERENCE = json.loads(
    (Path(__file__).parents[1] / "develop/parity/timeseries_ref.json").read_text()
)


def _data():
    x = pd.DataFrame(
        np.array([
            [-1.0, -0.7, -0.4, -0.1], [-0.7, -0.4, -0.1, 0.2],
            [-0.4, -0.1, 0.2, 0.5], [-0.1, 0.2, 0.5, 0.8],
            [0.2, 0.5, 0.8, 1.1], [0.5, 0.8, 1.1, 1.4],
            [0.8, 1.1, 1.4, 1.7], [1.1, 1.4, 1.7, 2.0],
        ], dtype=np.float32),
        columns=["t4", "t3", "t2", "t1"],
    )
    y = np.array([0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3], dtype=np.float32)
    return x, y


def _assert_reference(name, model):
    x, y = _data()
    model.fit(x, y)
    np.testing.assert_allclose(model.train_loss_, REFERENCE[name]["loss"], atol=1e-7)
    np.testing.assert_allclose(model.predict(x), REFERENCE[name]["one"], atol=1e-7)
    np.testing.assert_allclose(
        model.predict(x.iloc[[0]], steps_ahead=3), REFERENCE[name]["multi"], atol=1e-7
    )


def test_mlp_matches_frozen_legacy_one_and_multi_step():
    ref = REFERENCE["mlp"]
    _assert_reference("mlp", TorchTsMlp(
        input_size=4, hidden_sizes=[4], dropout=0.0, epochs=3,
        lr=0.01, batch_size=2, random_state=ref["seed"],
    ))


def test_conv1d_matches_frozen_legacy_one_and_multi_step():
    ref = REFERENCE["conv1d"]
    _assert_reference("conv1d", TsConv1D(
        input_size=4, conv_channels=[3], kernel_sizes=[2], strides=[1],
        dense_hidden_sizes=[4], epochs=3, lr=0.01, batch_size=2,
        random_state=ref["seed"],
    ))


def test_lstm_matches_frozen_legacy_one_and_multi_step():
    ref = REFERENCE["lstm"]
    _assert_reference("lstm", TsLstm(
        input_size=4, hidden_size=3, sequence_length=4, epochs=3,
        lr=0.01, batch_size=2, random_state=ref["seed"],
    ))
