import pytest
from src.optimizer.controller import (
    OptimizationController,
    OptimizationDecision,
    OptimizationAction,
)
from src.optimizer.resource_manager import ResourceManager, SystemState
from src.optimizer.adaptive_skip import AdaptiveFrameSkip
from src.optimizer.dynamic_resolution import DynamicResolution


def _make_controller(**rm_overrides):
    """Create a controller with stability=1 for deterministic single-sample tests."""
    rm_config = {"stability": {"consecutive_samples_required": 1}}
    if rm_overrides:
        rm_config.update(rm_overrides)
    rm = ResourceManager(config=rm_config)
    skip = AdaptiveFrameSkip(config={
        "min_skip": 0, "max_skip": 4,
        "recovery_step": 1, "minimum_hold_cycles": 2,
    })
    res = DynamicResolution(config={
        "enabled": True,
        "levels": [[640, 480], [512, 384], [480, 360], [320, 240]],
        "default_level": 0, "recovery_step": 1,
        "degradation_step": 1, "minimum_hold_cycles": 2,
    })
    return OptimizationController(
        resource_manager=rm,
        adaptive_skip=skip,
        dynamic_resolution=res,
    )


def test_low_load_decision():
    ctrl = _make_controller()
    decision = ctrl.evaluate({
        "cpu": 15.0, "memory": 20.0, "temperature": 40.0,
        "fps": 30.0, "inference_ms": 10.0,
    })
    assert decision.state == "LOW_LOAD"
    assert decision.action == OptimizationAction.RECOVER_WORKLOAD.value
    assert decision.frame_skip == 0
    assert decision.resolution == (640, 480)


def test_normal_decision():
    ctrl = _make_controller()
    decision = ctrl.evaluate({
        "cpu": 50.0, "memory": 50.0, "temperature": 55.0,
        "fps": 25.0, "inference_ms": 30.0,
    })
    assert decision.state == "NORMAL"
    assert decision.action == OptimizationAction.MAINTAIN.value


def test_high_load_decision():
    ctrl = _make_controller()
    decision = ctrl.evaluate({
        "cpu": 80.0, "memory": 50.0, "temperature": 55.0,
        "fps": 20.0, "inference_ms": 40.0,
    })
    assert decision.state == "HIGH_LOAD"
    assert decision.action == OptimizationAction.REDUCE_WORKLOAD.value
    assert decision.frame_skip >= 2
    assert decision.resolution[0] <= 640  # degraded or same


def test_critical_decision():
    ctrl = _make_controller()
    decision = ctrl.evaluate({
        "cpu": 95.0, "memory": 92.0, "temperature": 85.0,
        "fps": 5.0, "inference_ms": 250.0,
    })
    assert decision.state == "CRITICAL"
    assert decision.action == OptimizationAction.EMERGENCY_REDUCE.value
    assert decision.frame_skip == 4
    assert decision.resolution == (320, 240)


def test_frame_skip_integration():
    ctrl = _make_controller()
    # HIGH_LOAD should increase skip
    ctrl.evaluate({
        "cpu": 80.0, "memory": 50.0, "fps": 14.0, "inference_ms": 40.0,
    })
    assert ctrl.adaptive_skip.get_skip() >= 2


def test_resolution_integration():
    ctrl = _make_controller()
    # HIGH_LOAD should degrade resolution
    ctrl.evaluate({
        "cpu": 80.0, "memory": 50.0, "fps": 14.0, "inference_ms": 40.0,
    })
    assert ctrl.dynamic_resolution.get_level_index() >= 1


def test_multiple_simultaneous_problems():
    ctrl = _make_controller()
    decision = ctrl.evaluate({
        "cpu": 92.0, "memory": 91.0, "temperature": 82.0,
        "fps": 5.0, "inference_ms": 220.0,
    })
    assert decision.state == "CRITICAL"
    assert decision.frame_skip == 4
    assert decision.resolution == (320, 240)


def test_recovery_after_critical():
    ctrl = _make_controller()
    # Go critical
    ctrl.evaluate({
        "cpu": 95.0, "memory": 92.0, "temperature": 85.0,
        "fps": 5.0, "inference_ms": 250.0,
    })
    assert ctrl.adaptive_skip.get_skip() == 4
    assert ctrl.dynamic_resolution.get_level_index() == 3

    # Recover gradually — after several low-load cycles,
    # skip and resolution should start recovering
    for _ in range(6):
        ctrl.evaluate({
            "cpu": 15.0, "memory": 20.0, "temperature": 40.0,
            "fps": 30.0, "inference_ms": 10.0,
        })

    assert ctrl.adaptive_skip.get_skip() < 4
    assert ctrl.dynamic_resolution.get_level_index() < 3


def test_stable_decisions():
    ctrl = _make_controller()
    decisions = []
    for _ in range(5):
        d = ctrl.evaluate({
            "cpu": 50.0, "memory": 50.0, "temperature": 55.0,
            "fps": 25.0, "inference_ms": 30.0,
        })
        decisions.append(d.state)
    # All should be NORMAL
    assert all(s == "NORMAL" for s in decisions)


def test_missing_temperature():
    ctrl = _make_controller()
    decision = ctrl.evaluate({
        "cpu": 50.0, "memory": 50.0, "temperature": None,
        "fps": 25.0, "inference_ms": 30.0,
    })
    assert decision.thermal_state == "UNKNOWN"
    assert decision.state in ("NORMAL", "LOW_LOAD")


def test_invalid_metrics():
    ctrl = _make_controller()
    decision = ctrl.evaluate({})
    assert isinstance(decision, OptimizationDecision)
    assert isinstance(decision.state, str)


def test_controller_does_not_duplicate_threshold_logic():
    """Controller should delegate state classification entirely to ResourceManager."""
    ctrl = _make_controller()
    # Evaluate through controller
    decision = ctrl.evaluate({
        "cpu": 80.0, "memory": 50.0, "fps": 20.0, "inference_ms": 40.0,
    })
    # Evaluate directly through ResourceManager with same metrics
    rm_eval = ctrl.resource_manager.evaluate({
        "cpu": 80.0, "memory": 50.0, "fps": 20.0, "inference_ms": 40.0,
    })
    # Both should reach the same state classification
    assert decision.state == rm_eval.state.value


def test_decision_to_dict():
    ctrl = _make_controller()
    decision = ctrl.evaluate({
        "cpu": 50.0, "memory": 50.0, "temperature": 55.0,
        "fps": 25.0, "inference_ms": 30.0,
    })
    d = decision.to_dict()
    assert "state" in d
    assert "frame_skip" in d
    assert "resolution" in d
    assert "action" in d
    assert "reason" in d
    assert isinstance(d["resolution"], list)
