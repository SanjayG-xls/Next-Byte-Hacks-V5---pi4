import pytest
from src.optimizer.dynamic_resolution import DynamicResolution
from src.optimizer.resource_manager import SystemState


BASE_CONFIG = {
    "enabled": True,
    "levels": [[640, 480], [512, 384], [480, 360], [320, 240]],
    "default_level": 0,
    "recovery_step": 1,
    "degradation_step": 1,
    "minimum_hold_cycles": 3,
}


def _make(overrides=None):
    config = dict(BASE_CONFIG)
    if overrides:
        config.update(overrides)
    return DynamicResolution(config=config)


def test_initialization():
    dr = _make()
    assert dr.get_resolution() == (640, 480)
    assert dr.get_level_index() == 0
    assert dr.enabled is True


def test_highest_resolution_at_low_load():
    dr = _make()
    dr.update(SystemState.LOW_LOAD)
    assert dr.get_resolution() == (640, 480)


def test_resolution_stays_at_normal():
    dr = _make()
    dr.update(SystemState.NORMAL)
    assert dr.get_resolution() == (640, 480)


def test_resolution_degrades_under_high_load():
    dr = _make()
    res = dr.update(SystemState.HIGH_LOAD)
    assert dr.get_level_index() == 1
    assert res == (512, 384)


def test_aggressive_reduction_under_critical():
    dr = _make()
    res = dr.update(SystemState.CRITICAL)
    assert dr.get_level_index() == 3
    assert res == (320, 240)


def test_gradual_recovery():
    dr = _make({"minimum_hold_cycles": 2})
    # Degrade to lowest
    dr.update(SystemState.CRITICAL)
    assert dr.get_level_index() == 3

    # Recovery: hold for 2 cycles, then step by 1
    dr.update(SystemState.LOW_LOAD)  # hold 1
    assert dr.get_level_index() == 3

    dr.update(SystemState.LOW_LOAD)  # hold 2 -> recover 3->2
    assert dr.get_level_index() == 2

    dr.update(SystemState.LOW_LOAD)  # hold 1
    assert dr.get_level_index() == 2

    dr.update(SystemState.LOW_LOAD)  # hold 2 -> recover 2->1
    assert dr.get_level_index() == 1


def test_minimum_resolution_boundary():
    dr = _make()
    # Multiple CRITICAL updates should not go below last index
    dr.update(SystemState.CRITICAL)
    dr.update(SystemState.CRITICAL)
    dr.update(SystemState.CRITICAL)
    assert dr.get_level_index() == 3


def test_maximum_resolution_boundary():
    dr = _make({"minimum_hold_cycles": 0})
    # Already at level 0
    dr.update(SystemState.LOW_LOAD)
    assert dr.get_level_index() == 0


def test_hysteresis_no_flapping():
    dr = _make({"minimum_hold_cycles": 3})
    # Degrade
    dr.update(SystemState.HIGH_LOAD)
    assert dr.get_level_index() == 1

    # Single LOW_LOAD should NOT immediately recover
    dr.update(SystemState.LOW_LOAD)
    assert dr.get_level_index() == 1

    dr.update(SystemState.LOW_LOAD)
    assert dr.get_level_index() == 1


def test_stable_repeated_updates():
    dr = _make()
    # Repeated NORMAL updates at level 0 stay stable
    for _ in range(10):
        dr.update(SystemState.NORMAL)
    assert dr.get_level_index() == 0


def test_disabled_resolution():
    dr = _make({"enabled": False})
    dr.update(SystemState.CRITICAL)
    # Should NOT change when disabled
    assert dr.get_level_index() == 0
    assert dr.get_resolution() == (640, 480)


def test_missing_config():
    dr = DynamicResolution(config={})
    assert len(dr.levels) > 0
    assert dr.get_resolution() == dr.levels[0]


def test_invalid_levels_config():
    dr = DynamicResolution(config={"levels": ["bad", None, [0, -1]]})
    # Falls back to defaults since no valid levels parsed
    assert len(dr.levels) > 0
