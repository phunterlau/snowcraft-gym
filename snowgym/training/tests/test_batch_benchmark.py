from __future__ import annotations

from snowgym_training.benchmarks.throughput import (
    BENCHMARK_FORMAT,
    run_throughput_benchmark,
)


def test_batch_benchmark_reports_required_metrics() -> None:
    result = run_throughput_benchmark([1, 2], decisions=2)

    assert result["format"] == BENCHMARK_FORMAT
    for item in result["results"]:
        assert item["decisions"] == item["worlds"] * 2
        assert item["ticks"] == item["decisions"] * 6
        assert item["decisionsPerSecond"] > 0
        assert item["simulationTicksPerSecond"] > 0
        assert item["realTimeFactor"] > 0
        assert item["cpuUtilization"] > 0
        assert item["payloadBytes"] > 0
        assert 0 <= item["serializationShare"] <= 1
