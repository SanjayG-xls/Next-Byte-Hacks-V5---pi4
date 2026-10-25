from src.runtime.runtime import BytePi4Runtime


def test_runtime_creation():
    runtime = BytePi4Runtime()

    assert runtime is not None

    runtime.shutdown()


def test_runtime_start():
    runtime = BytePi4Runtime()

    runtime.start()

    assert runtime.running is True

    runtime.shutdown()


def test_runtime_stop():
    runtime = BytePi4Runtime()

    runtime.start()
    runtime.stop()

    assert runtime.running is False


def test_runtime_status():
    runtime = BytePi4Runtime()

    status = runtime.get_status()

    assert isinstance(status, dict)

    assert "running" in status
    assert "frame_count" in status
    assert "processed_frames" in status
    assert "system_state" in status
    assert "frame_skip" in status

    runtime.shutdown()


def test_runtime_reset():
    runtime = BytePi4Runtime()

    runtime.start()

    runtime.reset()

    assert runtime.frame_count == 0
    assert runtime.processed_frames == 0
    assert runtime.optimization_cycles == 0

    runtime.shutdown()