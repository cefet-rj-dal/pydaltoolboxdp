import numpy as np
import pandas as pd
import pytest
from pytspredit import TsRegSW

from pydaltoolboxdp import TorchTsMlp, TsConv1D, TsLstm


@pytest.fixture
def lagged_data():
    values = np.linspace(-1.0, 2.3, 15, dtype=np.float32)
    x = pd.DataFrame(
        [values[index : index + 4] for index in range(8)],
        columns=["t4", "t3", "t2", "t1"],
    )
    y = np.asarray([values[index + 4] for index in range(8)], dtype=np.float32)
    return x, y


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TorchTsMlp(
            input_size=4, hidden_sizes=[4], dropout=0.0,
            epochs=2, batch_size=2, random_state=101,
        ),
        lambda: TsConv1D(
            input_size=4, conv_channels=[3], kernel_sizes=[2], strides=[1],
            dense_hidden_sizes=[4], epochs=2, batch_size=2, random_state=103,
        ),
        lambda: TsLstm(
            input_size=4, hidden_size=3, sequence_length=4,
            epochs=2, batch_size=2, random_state=107,
        ),
    ],
)
def test_forecasters_implement_tsregsw_one_and_multi_step(factory, lagged_data):
    x, y = lagged_data
    model = factory().fit(x, y)

    assert isinstance(model, TsRegSW)
    assert model.predict(x).shape == (len(x),)
    assert model.predict(x.iloc[[0]], steps_ahead=3).shape == (3,)
    assert np.isfinite(model.predict(x.iloc[[0]], steps_ahead=3)).all()
    assert model.epochs_done == 2


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TorchTsMlp(
            input_size=4, hidden_sizes=[4], dropout=0.0,
            epochs=2, batch_size=2, random_state=109,
        ),
        lambda: TsConv1D(
            input_size=4, conv_channels=[3], kernel_sizes=[2], strides=[1],
            dense_hidden_sizes=[4], epochs=2, batch_size=2, random_state=113,
        ),
        lambda: TsLstm(
            input_size=4, hidden_size=3, sequence_length=4,
            epochs=2, batch_size=2, random_state=127,
        ),
    ],
)
def test_forecasters_are_reproducible(factory, lagged_data):
    x, y = lagged_data
    first = factory().fit(x, y)
    second = factory().fit(x, y)
    np.testing.assert_allclose(first.train_loss_, second.train_loss_, atol=1e-7)
    np.testing.assert_allclose(first.predict(x), second.predict(x), atol=1e-7)


def test_conv1d_validates_reshape(lagged_data):
    x, y = lagged_data
    with pytest.raises(ValueError, match="in_channels"):
        TsConv1D(input_size=4, in_channels=3, epochs=1).fit(x, y)


def test_lstm_validates_sequence_length(lagged_data):
    x, y = lagged_data
    with pytest.raises(ValueError, match="divisible"):
        TsLstm(input_size=4, sequence_length=3, epochs=1).fit(x, y)
