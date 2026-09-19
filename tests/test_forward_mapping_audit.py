import numpy as np

from scripts.audit_m52_forward_mapping import (
    centered_cosine,
    evaluate_network_oracle,
    network_power_basis,
)


def test_network_alpha_oracle_uses_fit_half_and_normalized_profiles() -> None:
    gain = np.asarray([[1.0, 0.5], [0.5, 1.0], [0.8, 0.8]])
    labels = np.asarray(["A", "B"])
    basis, names = network_power_basis(gain, labels)
    assert names == ["A", "B"]
    assert np.allclose(basis.mean(axis=0), 1.0)
    fit_profile = (basis[:, 0] / basis[:, 0].mean())[None]
    assert evaluate_network_oracle(basis, fit_profile, fit_profile)[0] < 1e-12
    unseen_profile = (basis[:, 1] / basis[:, 1].mean())[None]
    assert evaluate_network_oracle(basis, fit_profile, unseen_profile)[0] > 0.0


def test_centered_cosine_ignores_uniform_offset() -> None:
    first = np.asarray([1.0, 2.0, 3.0])
    second = first + 100.0
    assert np.isclose(centered_cosine(first, second), 1.0)
