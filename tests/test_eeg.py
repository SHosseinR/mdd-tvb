from pathlib import Path

import numpy as np

from mdd_tvb.config import load_config
from mdd_tvb.eeg import build_eeg_monitor, sensor_locations


def test_exact_tdbrain_montage() -> None:
    config = load_config(Path("configs/baseline.toml"))
    locations = sensor_locations(config.monitor)
    assert locations.shape == (26, 3)
    assert np.allclose(np.linalg.norm(locations, axis=1), 1.0)
    assert config.monitor.channels == (
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
        "FC3", "FCz", "FC4", "T7", "C3", "Cz", "C4", "T8",
        "CP3", "CPz", "CP4", "P7", "P3", "Pz", "P4", "P8",
        "O1", "Oz", "O2",
    )
    monitor, _ = build_eeg_monitor(config.monitor, 200, 2.0)
    assert monitor.obsnoise is None
