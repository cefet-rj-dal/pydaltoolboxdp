import numpy as np
import pytest
from pydaltoolbox import AutoencBaseE, AutoencBaseEd

from pydaltoolboxdp import AutoencDenoiseE, AutoencDenoiseEd, AutoencE, AutoencEd


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
