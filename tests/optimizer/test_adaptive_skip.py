import pytest
from src.optimizer.adaptive_skip import AdaptiveFrameSkip
from src.optimizer.resource_manager import SystemState


def test_adaptive_skip_initial():
    skip_ctrl = AdaptiveFrameSkip()
    assert skip_ctrl.get_skip() == 0
    assert skip_ctrl.should_run(0) is True
    assert skip_ctrl.should_run(1) is True


def test_adaptive_skip_should_run_pattern():
    skip_ctrl = AdaptiveFrameSkip(config={"min_skip": 0, "max_skip": 4})
    
    # Skip = 0 -> True for all frames
    for i in range(5):
        assert skip_ctrl.should_run(i) is True

    # Manually set skip = 1 -> True for 0, 2, 4, 6...
    skip_ctrl.current_skip = 1
    assert skip_ctrl.should_run(0) is True
    assert skip_ctrl.should_run(1) is False
    assert skip_ctrl.should_run(2) is True
    assert skip_ctrl.should_run(3) is False

    # Manually set skip = 2 -> True for 0, 3, 6, 9...
    skip_ctrl.current_skip = 2
    assert skip_ctrl.should_run(0) is True
    assert skip_ctrl.should_run(1) is False
    assert skip_ctrl.should_run(2) is False
    assert skip_ctrl.should_run(3) is True
    assert skip_ctrl.should_run(4) is False


def test_adaptive_skip_internal_counter():
    skip_ctrl = AdaptiveFrameSkip()
    skip_ctrl.current_skip = 1
    # Call without args: 0 (True), 1 (False), 2 (True), 3 (False)
    assert skip_ctrl.should_run() is True
    assert skip_ctrl.should_run() is False
    assert skip_ctrl.should_run() is True
    assert skip_ctrl.should_run() is False


def test_adaptive_skip_increase_under_high_load():
    skip_ctrl = AdaptiveFrameSkip()
    assert skip_ctrl.get_skip() == 0

    new_skip = skip_ctrl.update(SystemState.HIGH_LOAD)
    assert new_skip >= 2
    assert skip_ctrl.get_skip() == new_skip


def test_adaptive_skip_critical_load_max_bound():
    skip_ctrl = AdaptiveFrameSkip(config={"min_skip": 0, "max_skip": 4})
    new_skip = skip_ctrl.update(SystemState.CRITICAL)
    assert new_skip == 4

    # Extra update stays at max_skip
    new_skip_2 = skip_ctrl.update(SystemState.CRITICAL)
    assert new_skip_2 == 4


def test_adaptive_skip_gradual_recovery_and_hold():
    config = {
        "min_skip": 0,
        "max_skip": 4,
        "recovery_step": 1,
        "minimum_hold_cycles": 2,
    }
    skip_ctrl = AdaptiveFrameSkip(config=config)

    # Set to CRITICAL -> skip = 4
    skip_ctrl.update(SystemState.CRITICAL)
    assert skip_ctrl.get_skip() == 4

    # System returns to NORMAL.
    # Update 1: hold counter = 1 (< 2), skip stays 4
    skip_ctrl.update(SystemState.NORMAL)
    assert skip_ctrl.get_skip() == 4

    # Update 2: hold counter = 2 (>= 2), gradual recovery step = 1 -> skip decreases to 3
    skip_ctrl.update(SystemState.NORMAL)
    assert skip_ctrl.get_skip() == 3

    # Update 3: hold counter = 1 (< 2), skip stays 3
    skip_ctrl.update(SystemState.NORMAL)
    assert skip_ctrl.get_skip() == 3

    # Update 4: hold counter = 2 (>= 2), skip decreases to 2
    skip_ctrl.update(SystemState.NORMAL)
    assert skip_ctrl.get_skip() == 2


def test_adaptive_skip_min_bound():
    config = {
        "min_skip": 0,
        "max_skip": 4,
        "minimum_hold_cycles": 0,
    }
    skip_ctrl = AdaptiveFrameSkip(config=config)
    
    # Repeated NORMAL updates will not decrease below min_skip
    for _ in range(5):
        skip = skip_ctrl.update(SystemState.NORMAL)
        assert skip == 0
