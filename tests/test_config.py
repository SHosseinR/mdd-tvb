from pathlib import Path

from mdd_tvb.config import load_config


def test_baseline_config_contract() -> None:
    config = load_config(Path("configs/baseline.toml"))
    assert config.connectivity.expected_regions == 200
    assert len(config.monitor.channels) == 26
    assert config.simulation.monitor_period_ms == 2.0
    assert config.monitor.channels[3] == "F3"

