from prometheus_client import Gauge, Counter, start_http_server


class PrometheusExporter:

    def __init__(self, port: int = 8000):

        self.port = port

        self.cpu_usage = Gauge(
            "bytepi4_cpu_usage_percent",
            "Current CPU utilization percentage",
        )

        self.memory_usage = Gauge(
            "bytepi4_memory_usage_percent",
            "Current memory utilization percentage",
        )

        self.temperature = Gauge(
            "bytepi4_temperature_celsius",
            "Current system temperature in Celsius",
        )

        self.fps = Gauge(
            "bytepi4_fps",
            "Current frames per second",
        )

        self.latency = Gauge(
            "bytepi4_latency_ms",
            "Current processing latency in milliseconds",
        )

        self.frame_skip = Gauge(
            "bytepi4_frame_skip",
            "Current adaptive frame skip value",
        )

        self.optimization_cycles = Counter(
            "bytepi4_optimization_cycles_total",
            "Total number of optimization cycles",
        )

        self.total_frames = Counter(
            "bytepi4_frames_total",
            "Total number of frames received",
        )

        self.processed_frames = Counter(
            "bytepi4_processed_frames_total",
            "Total number of processed frames",
        )

        self.system_state = Gauge(
            "bytepi4_system_state",
            "Current bytepi4 optimization state",
            ["state"],
        )

        self.runtime_running = Gauge(
            "bytepi4_runtime_running",
            "Whether bytepi4 runtime is running",
        )

        self.resolution_width = Gauge(
            "bytepi4_resolution_width",
            "Current processing resolution width",
        )

        self.resolution_height = Gauge(
            "bytepi4_resolution_height",
            "Current processing resolution height",
        )

        self.uptime = Gauge(
            "bytepi4_uptime_seconds",
            "bytepi4 runtime uptime in seconds",
        )

        self._last_total_frames = 0
        self._last_processed_frames = 0
        self._last_optimization_cycles = 0

    def start(self):

        start_http_server(self.port)

        print(
            f"bytepi4 Prometheus exporter started on port {self.port}"
        )

        print(
            f"Metrics endpoint: http://localhost:{self.port}/metrics"
        )

    def update(self, profiler, runtime):

        metrics = profiler.get_metrics()

        cpu = metrics.get("cpu")
        memory = metrics.get("memory")
        temperature = metrics.get("temperature")
        fps = metrics.get("fps")
        latency = metrics.get("inference_ms")

        if cpu is not None:
            self.cpu_usage.set(cpu)

        if memory is not None:
            self.memory_usage.set(memory)

        if temperature is not None:
            self.temperature.set(temperature)

        if fps is not None:
            self.fps.set(fps)

        if latency is not None:
            self.latency.set(latency)

        status = runtime.get_status()

        self.runtime_running.set(
            1 if status.get("running", False) else 0
        )

        self.frame_skip.set(
            status.get("frame_skip", 0)
        )

        self.uptime.set(
            status.get("uptime_seconds", 0)
        )

        total_frames = status.get(
            "frame_count",
            0,
        )

        processed_frames = status.get(
            "processed_frames",
            0,
        )

        optimization_cycles = status.get(
            "optimization_cycles",
            0,
        )

        if total_frames >= self._last_total_frames:

            self.total_frames.inc(
                total_frames - self._last_total_frames
            )

        else:

            self._last_total_frames = 0

            self.total_frames.inc(
                total_frames
            )

        self._last_total_frames = total_frames

        if processed_frames >= self._last_processed_frames:

            self.processed_frames.inc(
                processed_frames - self._last_processed_frames
            )

        else:

            self._last_processed_frames = 0

            self.processed_frames.inc(
                processed_frames
            )

        self._last_processed_frames = processed_frames

        if optimization_cycles >= self._last_optimization_cycles:

            self.optimization_cycles.inc(
                optimization_cycles - self._last_optimization_cycles
            )

        else:

            self._last_optimization_cycles = 0

            self.optimization_cycles.inc(
                optimization_cycles
            )

        self._last_optimization_cycles = optimization_cycles

        state = status.get(
            "system_state",
            "UNKNOWN",
        )

        known_states = [
            "UNKNOWN",
            "LOW_LOAD",
            "NORMAL",
            "HIGH_LOAD",
            "CRITICAL",
        ]

        for known_state in known_states:

            self.system_state.labels(
                state=known_state
            ).set(
                1 if state == known_state else 0
            )

        resolution = status.get("resolution")

        if resolution is not None:

            if isinstance(
                resolution,
                (tuple, list),
            ) and len(resolution) >= 2:

                self.resolution_width.set(
                    float(resolution[0])
                )

                self.resolution_height.set(
                    float(resolution[1])
                )

            elif isinstance(
                resolution,
                dict,
            ):

                width = resolution.get("width")
                height = resolution.get("height")

                if width is not None:
                    self.resolution_width.set(
                        float(width)
                    )

                if height is not None:
                    self.resolution_height.set(
                        float(height)
                    )