import pytest
from src.optimizer.resource_manager import (
    ResourceManager,
    SystemState,
    ThermalState,
    ResourceEvaluation,
)


def test_resource_manager_low_load():
    rm = ResourceManager(config={"stability": {"consecutive_samples_required": 1}})
    metrics = {
        "cpu": 15.0,
        "memory": 20.0,
        "temperature": 40.0,
        "fps": 30.0,
        "inference_ms": 10.0,
    }
    eval_res = rm.evaluate(metrics)
    assert eval_res.state == SystemState.LOW_LOAD
    assert eval_res.thermal_state == ThermalState.NORMAL
    assert rm.get_state() == SystemState.LOW_LOAD


def test_resource_manager_normal_load():
    rm = ResourceManager(config={"stability": {"consecutive_samples_required": 1}})
    metrics = {
        "cpu": 50.0,
        "memory": 50.0,
        "temperature": 55.0,
        "fps": 25.0,
        "inference_ms": 30.0,
    }
    eval_res = rm.evaluate(metrics)
    assert eval_res.state == SystemState.NORMAL
    assert eval_res.thermal_state == ThermalState.NORMAL


def test_resource_manager_high_load_cpu():
    rm = ResourceManager(config={"stability": {"consecutive_samples_required": 1}})
    metrics = {
        "cpu": 80.0,
        "memory": 50.0,
        "temperature": 55.0,
        "fps": 20.0,
        "inference_ms": 40.0,
    }
    eval_res = rm.evaluate(metrics)
    assert eval_res.state == SystemState.HIGH_LOAD
    assert "CPU utilization is high" in eval_res.reason


def test_resource_manager_high_load_ram():
    rm = ResourceManager(config={"stability": {"consecutive_samples_required": 1}})
    metrics = {
        "cpu": 40.0,
        "memory": 82.0,
        "temperature": 55.0,
        "fps": 20.0,
        "inference_ms": 40.0,
    }
    eval_res = rm.evaluate(metrics)
    assert eval_res.state == SystemState.HIGH_LOAD
    assert "Memory usage is high" in eval_res.reason


def test_resource_manager_high_load_latency():
    rm = ResourceManager(config={"stability": {"consecutive_samples_required": 1}})
    metrics = {
        "cpu": 40.0,
        "memory": 50.0,
        "temperature": 55.0,
        "fps": 20.0,
        "inference_ms": 120.0,
    }
    eval_res = rm.evaluate(metrics)
    assert eval_res.state == SystemState.HIGH_LOAD
    assert "Latency is high" in eval_res.reason


def test_resource_manager_low_fps():
    rm = ResourceManager(config={"stability": {"consecutive_samples_required": 1}})
    metrics = {
        "cpu": 40.0,
        "memory": 50.0,
        "temperature": 55.0,
        "fps": 12.0,
        "inference_ms": 30.0,
    }
    eval_res = rm.evaluate(metrics)
    assert eval_res.state == SystemState.HIGH_LOAD
    assert "FPS is below target" in eval_res.reason


def test_resource_manager_critical_load_cpu():
    rm = ResourceManager(config={"stability": {"consecutive_samples_required": 1}})
    metrics = {
        "cpu": 95.0,
        "memory": 50.0,
        "temperature": 60.0,
        "fps": 10.0,
        "inference_ms": 50.0,
    }
    eval_res = rm.evaluate(metrics)
    assert eval_res.state == SystemState.CRITICAL
    assert "CPU utilization is critical" in eval_res.reason


def test_resource_manager_thermal_conditions():
    rm = ResourceManager(config={"stability": {"consecutive_samples_required": 1}})

    # High temperature
    high_temp_metrics = {
        "cpu": 40.0,
        "memory": 40.0,
        "temperature": 72.0,
        "fps": 25.0,
        "inference_ms": 20.0,
    }
    eval_high = rm.evaluate(high_temp_metrics)
    assert eval_high.state == SystemState.HIGH_LOAD
    assert eval_high.thermal_state == ThermalState.HIGH
    assert "Temperature is high" in eval_high.reason

    # Critical temperature
    crit_temp_metrics = {
        "cpu": 40.0,
        "memory": 40.0,
        "temperature": 85.0,
        "fps": 25.0,
        "inference_ms": 20.0,
    }
    eval_crit = rm.evaluate(crit_temp_metrics)
    assert eval_crit.state == SystemState.CRITICAL
    assert eval_crit.thermal_state == ThermalState.CRITICAL
    assert "Temperature is critical" in eval_crit.reason

    # Missing temperature (laptop without temp sensor)
    no_temp_metrics = {
        "cpu": 20.0,
        "memory": 20.0,
        "temperature": None,
        "fps": 30.0,
        "inference_ms": 10.0,
    }
    eval_no_temp = rm.evaluate(no_temp_metrics)
    assert eval_no_temp.state == SystemState.LOW_LOAD
    assert eval_no_temp.thermal_state == ThermalState.UNKNOWN


def test_resource_manager_multiple_simultaneous_problems():
    rm = ResourceManager(config={"stability": {"consecutive_samples_required": 1}})
    metrics = {
        "cpu": 88.0,
        "memory": 85.0,
        "temperature": 75.0,
        "fps": 10.0,
        "inference_ms": 150.0,
    }
    eval_res = rm.evaluate(metrics)
    assert eval_res.state == SystemState.HIGH_LOAD
    assert "CPU utilization is high" in eval_res.reason
    assert "Memory usage is high" in eval_res.reason
    assert "Temperature is high" in eval_res.reason


def test_resource_manager_stability_anti_flapping():
    # Require 2 consecutive samples for transition
    rm = ResourceManager(config={"stability": {"consecutive_samples_required": 2}})
    
    normal_metrics = {"cpu": 50.0, "memory": 50.0, "fps": 25.0, "inference_ms": 30.0}
    rm.evaluate(normal_metrics)
    assert rm.get_state() == SystemState.NORMAL

    high_metrics = {"cpu": 80.0, "memory": 50.0, "fps": 20.0, "inference_ms": 40.0}
    
    # 1st high sample: state remains NORMAL because 2 consecutive samples are required
    eval1 = rm.evaluate(high_metrics)
    assert eval1.state == SystemState.NORMAL

    # 2nd high sample: state transitions to HIGH_LOAD
    eval2 = rm.evaluate(high_metrics)
    assert eval2.state == SystemState.HIGH_LOAD


def test_resource_manager_to_dict():
    rm = ResourceManager(config={"stability": {"consecutive_samples_required": 1}})
    metrics = {"cpu": 50.0, "memory": 50.0, "temperature": 55.0, "fps": 25.0, "inference_ms": 30.0}
    eval_res = rm.evaluate(metrics)
    d = eval_res.to_dict()
    assert d["state"] == "NORMAL"
    assert d["cpu"] == 50.0
    assert d["thermal_state"] == "NORMAL"
    assert "reason" in d
