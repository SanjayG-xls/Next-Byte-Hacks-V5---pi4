import pytest
from src.profiler.profiler import Profiler
from src.optimizer.resource_manager import ResourceManager, SystemState
from src.optimizer.adaptive_skip import AdaptiveFrameSkip


def test_profiler_resource_manager_adaptive_skip_pipeline():
    # 1. Instantiate components
    profiler = Profiler()
    rm = ResourceManager(config={"stability": {"consecutive_samples_required": 1}})
    skip_ctrl = AdaptiveFrameSkip(config={"minimum_hold_cycles": 1, "recovery_step": 1})

    # 2. Get metrics from Profiler
    metrics = profiler.get_metrics()
    assert isinstance(metrics, dict)
    assert "cpu" in metrics
    assert "memory" in metrics
    assert "fps" in metrics

    # 3. Evaluate system state with ResourceManager
    evaluation = rm.evaluate(metrics)
    assert isinstance(evaluation.state, SystemState)
    assert evaluation.reason is not None

    # 4. Update AdaptiveFrameSkip decision
    skip_level = skip_ctrl.update(evaluation.state, fps=evaluation.fps, latency=evaluation.latency_ms)
    assert isinstance(skip_level, int)
    assert 0 <= skip_level <= 4

    # 5. Check frame execution decision
    should_run_frame0 = skip_ctrl.should_run(0)
    assert isinstance(should_run_frame0, bool)
