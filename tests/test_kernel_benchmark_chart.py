from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

from benchmarks.kernel_benchmark_chart import (
    DEFAULT_DATA,
    DEFAULT_OUTPUT,
    generate,
    load_data,
)


def test_kernel_benchmark_data_matches_latency_reductions() -> None:
    data = load_data(DEFAULT_DATA)
    measurements = {item["operation"]: item for item in data["measurements"]}

    assert measurements["Forward"]["latency_reduction_percent"] == 17.738713181176912
    assert measurements["Backward"]["latency_reduction_percent"] == 16.987145180006603
    assert "zero-initializations" in measurements["Backward"]["scope"]


def test_committed_kernel_chart_is_deterministic(tmp_path: Path) -> None:
    generated = tmp_path / "kernel-benchmark.svg"
    source_sha256 = generate(DEFAULT_DATA, generated)

    assert generated.read_bytes() == DEFAULT_OUTPUT.read_bytes()
    assert source_sha256 == hashlib.sha256(DEFAULT_DATA.read_bytes()).hexdigest()
    svg = generated.read_text(encoding="utf-8")
    assert source_sha256 in svg
    assert "17.7% lower latency" in svg
    assert "17.0% lower latency" in svg
    ET.fromstring(svg)
