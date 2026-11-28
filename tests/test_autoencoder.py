import numpy as np
import pytest
import torch
from pydaltoolbox import AutoencBaseE, AutoencBaseEd

from pydaltoolboxdp import (
    AutoencAdvE,
    AutoencAdvEd,
    AutoencConvE,
    AutoencConvEd,
    AutoencDenoiseE,
    AutoencDenoiseEd,
    AutoencE,
    AutoencEd,
    AutoencLstmE,
    AutoencLstmEd,
    AutoencStackedE,
    AutoencStackedEd,
    AutoencVariationalE,
    AutoencVariationalEd,
)


@pytest.fixture
def data():
    rng = np.random.default_rng(7)
    return rng.normal(size=(24, 4)).astype(np.float32)


def test_autoenc_e_implements_contract_and_encodes(data):
    model = AutoencE(4, 2, epochs=3, batch_size=6, random_state=11).fit(data)
    encoded = model.transform(data)

    assert isinstance(model, AutoencBaseE)
    assert encoded.shape == (24, 2)
    assert len(model.train_loss_) == 3
    assert model.val_loss_ == []


def test_autoenc_ed_implements_contract_and_reconstructs(data):
    model = AutoencEd(4, 2, epochs=3, batch_size=6, random_state=11).fit(data)
    reconstructed = model.transform(data)

    assert isinstance(model, AutoencBaseEd)
    assert reconstructed.shape == data.shape
    assert np.isfinite(reconstructed).all()


def test_seeded_training_is_reproducible(data):
    first = AutoencEd(4, 2, epochs=3, batch_size=6, random_state=19).fit(data)
    second = AutoencEd(4, 2, epochs=3, batch_size=6, random_state=19).fit(data)

    np.testing.assert_allclose(first.transform(data), second.transform(data), atol=1e-7)
    np.testing.assert_allclose(first.train_loss_, second.train_loss_, atol=1e-7)


def test_validation_and_early_stopping_contract(data):
    model = AutoencEd(
        4,
        2,
        epochs=8,
        batch_size=4,
        stopping_rule="patience",
        patience=1,
        random_state=3,
    ).fit(data)

    assert model.val_loss_
    assert model.epochs_done_ <= 8


def test_fit_epoch_override_supports_adaptive_consumers(data):
    model = AutoencEd(4, 2, epochs=10, random_state=5).fit(data, epochs=2)
    assert model.epochs_done_ == 2


def test_invalid_feature_count_is_rejected(data):
    model = AutoencE(5, 2, epochs=1)
    with pytest.raises(ValueError, match="expected 5 features"):
        model.fit(data)


def test_denoising_encoder_implements_contract(data):
    model = AutoencDenoiseE(
        4, 2, noise_factor=0.2, epochs=3, batch_size=6, random_state=13
    ).fit(data)

    assert isinstance(model, AutoencBaseE)
    assert model.transform(data).shape == (24, 2)
    assert len(model.train_loss_) == 3


def test_denoising_reconstruction_is_seeded(data):
    first = AutoencDenoiseEd(
        4, 2, noise_factor=0.2, epochs=3, batch_size=6, random_state=17
    ).fit(data)
    second = AutoencDenoiseEd(
        4, 2, noise_factor=0.2, epochs=3, batch_size=6, random_state=17
    ).fit(data)

    assert isinstance(first, AutoencBaseEd)
    np.testing.assert_allclose(first.transform(data), second.transform(data), atol=1e-7)
    np.testing.assert_allclose(first.train_loss_, second.train_loss_, atol=1e-7)


def test_negative_noise_factor_is_rejected():
    with pytest.raises(ValueError, match="noise_factor must be non-negative"):
        AutoencDenoiseE(4, 2, noise_factor=-0.1)


def test_convolutional_encoder_implements_contract(data):
    model = AutoencConvE(
        4, 2, hidden_channels=3, bottleneck_channels=4,
        epochs=3, batch_size=6, random_state=29,
    ).fit(data)

    assert isinstance(model, AutoencBaseE)
    assert model.transform(data).shape == (24, 2)
    assert len(model.train_loss_) == 3


def test_convolutional_reconstruction_is_seeded(data):
    options = {
        "hidden_channels": 3,
        "bottleneck_channels": 4,
        "epochs": 3,
        "batch_size": 6,
        "random_state": 31,
    }
    first = AutoencConvEd(4, 2, **options).fit(data)
    second = AutoencConvEd(4, 2, **options).fit(data)

    assert isinstance(first, AutoencBaseEd)
    assert first.transform(data).shape == data.shape
    np.testing.assert_allclose(first.transform(data), second.transform(data), atol=1e-7)
    np.testing.assert_allclose(first.train_loss_, second.train_loss_, atol=1e-7)


def test_convolutional_kernel_must_be_positive_and_odd():
    with pytest.raises(ValueError, match="positive odd"):
        AutoencConvE(4, 2, kernel_size=2)


def test_lstm_encoder_implements_contract(data):
    model = AutoencLstmE(
        4, 2, sequence_length=2, lstm_hidden_size=3,
        epochs=3, batch_size=6, random_state=41,
    ).fit(data)

    assert isinstance(model, AutoencBaseE)
    assert model.transform(data).shape == (24, 2)
    assert len(model.train_loss_) == 3


def test_lstm_reconstruction_is_seeded(data):
    options = {
        "sequence_length": 2,
        "lstm_hidden_size": 3,
        "epochs": 3,
        "batch_size": 6,
        "random_state": 43,
    }
    first = AutoencLstmEd(4, 2, **options).fit(data)
    second = AutoencLstmEd(4, 2, **options).fit(data)

    assert isinstance(first, AutoencBaseEd)
    assert first.transform(data).shape == data.shape
    np.testing.assert_allclose(first.transform(data), second.transform(data), atol=1e-7)
    np.testing.assert_allclose(first.train_loss_, second.train_loss_, atol=1e-7)


def test_lstm_sequence_length_must_divide_input_size():
    with pytest.raises(ValueError, match="divisible"):
        AutoencLstmE(5, 2, sequence_length=2)


def test_stacked_encoder_implements_contract(data):
    model = AutoencStackedE(
        4, 2, encoding_sizes=[3, 2], encoder_hidden_sizes=[5],
        epochs=2, batch_size=6, random_state=61,
    ).fit(data)

    assert isinstance(model, AutoencBaseE)
    assert model.transform(data).shape == (24, 2)
    assert model.epochs_done_ == 4


def test_stacked_reconstruction_is_seeded(data):
    options = {
        "encoding_sizes": [3, 2],
        "encoder_hidden_sizes": [5],
        "epochs": 2,
        "batch_size": 6,
        "random_state": 67,
    }
    first = AutoencStackedEd(4, 2, **options).fit(data)
    second = AutoencStackedEd(4, 2, **options).fit(data)

    assert isinstance(first, AutoencBaseEd)
    assert first.transform(data).shape == data.shape
    np.testing.assert_allclose(first.transform(data), second.transform(data), atol=1e-7)
    np.testing.assert_allclose(first.train_loss_, second.train_loss_, atol=1e-7)


def test_stacked_stage_specific_parameters_must_match_stage_count():
    with pytest.raises(ValueError, match="match the number of stages"):
        AutoencStackedE(
            4,
            2,
            encoding_sizes=[3, 2],
            encoder_hidden_sizes=[[5]],
        )


def test_variational_encoder_returns_distribution_parameters(data):
    bounded = 1 / (1 + np.exp(-data))
    model = AutoencVariationalE(
        4, 2, encoder_hidden_sizes=[5], decoder_hidden_sizes=[5],
        epochs=2, batch_size=6, random_state=83,
    ).fit(bounded)

    assert isinstance(model, AutoencBaseE)
    assert model.transform(bounded).shape == (24, 4)


def test_variational_reconstruction_is_seeded(data):
    bounded = 1 / (1 + np.exp(-data))
    options = {
        "encoder_hidden_sizes": [5], "decoder_hidden_sizes": [5],
        "epochs": 2, "batch_size": 6, "random_state": 89,
    }
    first = AutoencVariationalEd(4, 2, **options).fit(bounded)
    second = AutoencVariationalEd(4, 2, **options).fit(bounded)
    torch.manual_seed(97)
    first_result = first.transform(bounded)
    torch.manual_seed(97)
    second_result = second.transform(bounded)

    assert isinstance(first, AutoencBaseEd)
    np.testing.assert_allclose(first_result, second_result, atol=1e-7)
    np.testing.assert_allclose(first.train_loss_, second.train_loss_, atol=1e-7)


def test_variational_reconstruction_loss_is_validated():
    with pytest.raises(ValueError, match="reconstruction_loss"):
        AutoencVariationalE(4, 2, reconstruction_loss="mae")


def test_adversarial_encoder_implements_contract(data):
    bounded = 1 / (1 + np.exp(-data))
    model = AutoencAdvE(
        4, 2, encoder_hidden_sizes=[5], decoder_hidden_sizes=[5],
        discriminator_hidden_sizes=[5], dropout=0.1,
        epochs=2, batch_size=6, random_state=103,
    ).fit(bounded)
    assert isinstance(model, AutoencBaseE)
    assert model.transform(bounded).shape == (24, 2)


def test_adversarial_reconstruction_is_seeded(data):
    bounded = 1 / (1 + np.exp(-data))
    options = {
        "encoder_hidden_sizes": [5], "decoder_hidden_sizes": [5],
        "discriminator_hidden_sizes": [5], "dropout": 0.1,
        "epochs": 2, "batch_size": 6, "random_state": 107,
    }
    first = AutoencAdvEd(4, 2, **options).fit(bounded)
    second = AutoencAdvEd(4, 2, **options).fit(bounded)
    assert isinstance(first, AutoencBaseEd)
    np.testing.assert_allclose(first.transform(bounded), second.transform(bounded), atol=1e-7)
    np.testing.assert_allclose(first.train_loss_, second.train_loss_, atol=1e-7)
