from pathlib import Path

import numpy as np

from mdd_tvb.config import load_config
from mdd_tvb.connectome import load_connectome, transform_weights


def test_region_sum_scaling_separates_global_coupling() -> None:
    raw = np.array([[0.0, 1.0, 3.0], [1.0, 0.0, 2.0], [3.0, 2.0, 0.0]])
    transformed, weights = transform_weights(raw, "log1p", "region_sum")
    assert transformed.shape == raw.shape
    assert np.isclose(weights.sum(axis=1).max(), 1.0)
    assert np.allclose(weights, weights.T)


def test_project_connectome_audit() -> None:
    config = load_config(Path("configs/baseline.toml"))
    bundle = load_connectome(config.paths, config.connectivity)
    assert bundle.weights.shape == (200, 200)
    assert bundle.audit["connected_components"] == 1
    assert bundle.audit["undirected_edges"] == 13861
    assert np.array_equal(bundle.weights > 0, bundle.tract_lengths > 0)
    assert np.isclose(bundle.weights.sum(axis=1).max(), 1.0)

