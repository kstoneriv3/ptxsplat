from __future__ import annotations

import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest

from benchmarks.performance_history import (
    ValidationError,
    build_report_spec,
    paired_summary,
    promoted_frontier,
    sequential_ablation,
    validate_study,
)
from benchmarks.performance_report import main, render_assets, write_report


BASELINE_COMMIT = "a" * 40
ENVIRONMENT = {
    "environment_id": "synthetic-rtx5090",
    "gpu_name": "Synthetic RTX 5090",
    "gpu_uuid": "GPU-synthetic",
    "container_image": "360-video-gs-dev@sha256:" + "b" * 64,
    "driver_version": "synthetic-driver",
    "cuda_version": "12.8",
    "pytorch_version": "2.9.1",
    "build_flags": ["-O3", "--use_fast_math", "sm_120"],
    "sm_clock_mhz": 2400,
    "memory_clock_mhz": 14000,
}
PROTOCOL = {
    "evaluator_commit": "e" * 40,
    "evaluator_sha256": "f" * 64,
    "startup_compile_excluded": True,
    "seeds": [42, 43, 44, 45, 46],
    "pair_order": [
        "baseline-first",
        "candidate-first",
        "baseline-first",
        "candidate-first",
        "baseline-first",
    ],
    "warmup_iterations": 20,
    "measured_iterations": 100,
    "rounds": 5,
    "training_steps": 7000,
    "progress_scene_id": "bonsai",
    "progress_scope": "forward_backward",
}
DATASETS = [
    {
        "scene_id": "bonsai",
        "dataset": "mipnerf360",
        "sha256": "1" * 64,
        "data_factor": 2,
    },
    {
        "scene_id": "garden",
        "dataset": "mipnerf360",
        "sha256": "2" * 64,
        "data_factor": 4,
    },
]

BASE_SAMPLES = {
    ("bonsai", "isolated_forward"): [10.0, 10.2, 9.8, 10.1, 9.9],
    ("bonsai", "isolated_backward"): [20.0, 20.4, 19.6, 20.2, 19.8],
    ("bonsai", "forward_backward"): [30.0, 30.6, 29.4, 30.3, 29.7],
    ("bonsai", "training"): [700.0, 714.0, 686.0, 707.0, 693.0],
    ("garden", "isolated_forward"): [12.0, 12.24, 11.76, 12.12, 11.88],
    ("garden", "isolated_backward"): [24.0, 24.48, 23.52, 24.24, 23.76],
    ("garden", "forward_backward"): [36.0, 36.72, 35.28, 36.36, 35.64],
    ("garden", "training"): [800.0, 816.0, 784.0, 808.0, 792.0],
}


def _measurement(scene: str, scope: str, factor: float) -> dict[str, Any]:
    baseline = BASE_SAMPLES[(scene, scope)]
    return {
        "scene_id": scene,
        "scope": scope,
        "unit": "s" if scope == "training" else "ms",
        "baseline_samples": baseline,
        "candidate_samples": [value * factor for value in baseline],
    }


def _record(
    experiment_id: str,
    attempt: int,
    commit_digit: str,
    factor: float,
    *,
    status: str,
    publication: bool,
    promoted: bool,
    ablation_order: int | None = None,
    baseline: bool = False,
) -> dict[str, Any]:
    candidate = {
        "commit": BASELINE_COMMIT if baseline else commit_digit * 40,
        "label": "Reference" if baseline else f"Candidate {commit_digit}",
    }
    baseline_side = {"commit": BASELINE_COMMIT, "label": "Reference"}
    measurements = []
    if status != "crash":
        scopes = (
            ("isolated_forward", "isolated_backward", "forward_backward", "training")
            if publication
            else ("forward_backward",)
        )
        scenes = ("bonsai", "garden") if publication else ("bonsai",)
        measurements = [
            _measurement(scene, scope, factor) for scene in scenes for scope in scopes
        ]
    record: dict[str, Any] = {
        "schema_version": 1,
        "record_type": "ptxsplat.performance_history.result",
        "study_id": "synthetic-study",
        "experiment_id": experiment_id,
        "attempt_index": attempt,
        "timestamp_utc": f"2026-07-16T12:{attempt:02d}:00Z",
        "status": status,
        "description": f"Synthetic attempt {attempt}",
        "candidate": candidate,
        "baseline": baseline_side,
        "comparison_kind": "baseline" if baseline else "historical",
        "publication": publication,
        "promoted": promoted,
        "environment": copy.deepcopy(ENVIRONMENT),
        "protocol": copy.deepcopy(PROTOCOL),
        "datasets": copy.deepcopy(DATASETS),
        "correctness": {
            "passed": status != "crash",
            "gate": "synthetic correctness gate",
            "artifact_sha256": "c" * 64 if status != "crash" else None,
        },
        "measurements": measurements,
    }
    if ablation_order is not None:
        record["ablation"] = {
            "group_id": "promoted-history",
            "order": ablation_order,
            "component": (
                "Reference baseline"
                if ablation_order == 0
                else f"Synthetic optimization {ablation_order}"
            ),
            "method": "historical_checkpoint",
        }
    return record


@pytest.fixture
def synthetic_records() -> list[dict[str, Any]]:
    return [
        _record(
            "baseline",
            0,
            "a",
            1.0,
            status="keep",
            publication=True,
            promoted=True,
            ablation_order=0,
            baseline=True,
        ),
        _record(
            "optimization-one",
            1,
            "3",
            0.9,
            status="keep",
            publication=True,
            promoted=True,
            ablation_order=1,
        ),
        _record(
            "discarded-attempt",
            2,
            "4",
            1.05,
            status="discard",
            publication=False,
            promoted=False,
        ),
        _record(
            "crashed-attempt",
            3,
            "5",
            1.0,
            status="crash",
            publication=False,
            promoted=False,
        ),
        _record(
            "optimization-two",
            4,
            "6",
            0.8,
            status="keep",
            publication=True,
            promoted=True,
            ablation_order=2,
        ),
    ]


def test_schema_validation_and_publication_completeness(
    synthetic_records: list[dict[str, Any]],
) -> None:
    assert len(validate_study(synthetic_records)) == 5

    incomplete = copy.deepcopy(synthetic_records)
    incomplete[1]["measurements"] = [
        item
        for item in incomplete[1]["measurements"]
        if not (item["scene_id"] == "garden" and item["scope"] == "training")
    ]
    with pytest.raises(ValidationError, match="publication matrix incomplete"):
        validate_study(incomplete)

    unknown = copy.deepcopy(synthetic_records)
    unknown[0]["unverified_claim"] = True
    with pytest.raises(ValidationError, match="unknown fields"):
        validate_study(unknown)


def test_paired_baseline_math_uses_aligned_samples() -> None:
    summary = paired_summary(
        _measurement("bonsai", "forward_backward", 0.8),
        bootstrap_samples=300,
        seed_key="math",
    )
    assert summary["speedup_x"] == pytest.approx(1.25)
    assert summary["duration_reduction_percent"] == pytest.approx(20.0)
    assert summary["speedup_ci95_x"] == pytest.approx([1.25, 1.25])
    assert summary["candidate_median_ci95"][0] < summary["candidate_median_ci95"][1]


def test_promoted_frontier_retains_discard_and_crash_attempts(
    synthetic_records: list[dict[str, Any]],
) -> None:
    progress = promoted_frontier(synthetic_records, bootstrap_samples=200)
    assert [item["status"] for item in progress] == [
        "keep",
        "keep",
        "discard",
        "crash",
        "keep",
    ]
    assert progress[3]["score"] is None
    assert progress[2]["frontier_speedup_x"] == pytest.approx(1.0 / 0.9)
    assert progress[4]["frontier_speedup_x"] == pytest.approx(1.25)


def test_sequential_history_contributions_are_ordered(
    synthetic_records: list[dict[str, Any]],
) -> None:
    result = sequential_ablation(synthetic_records, bootstrap_samples=200)
    assert result["method"] == "historical_checkpoint"
    assert [
        step["contribution_percentage_points"] for step in result["steps"]
    ] == pytest.approx([0.0, 10.0, 10.0])


def test_report_spec_and_svg_outputs_are_deterministic(
    synthetic_records: list[dict[str, Any]], tmp_path: Path
) -> None:
    first_spec = build_report_spec(synthetic_records, bootstrap_samples=250)
    second_spec = build_report_spec(
        list(reversed(synthetic_records)), bootstrap_samples=250
    )
    assert first_spec == second_spec
    assert render_assets(first_spec) == render_assets(second_spec)

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    write_report(first_spec, first_dir)
    write_report(second_spec, second_dir)
    first_files = sorted(path.name for path in first_dir.iterdir())
    assert first_files == [
        "REPORT.md",
        "operation-breakdown.svg",
        "optimization-progress.svg",
        "per-scene-speedup.svg",
        "report-spec.json",
        "sequential-history-waterfall.svg",
        "training-time-throughput.svg",
    ]
    for name in first_files:
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()
        if name.endswith(".svg"):
            ET.parse(first_dir / name)
    svg = (first_dir / "per-scene-speedup.svg").read_text(encoding="utf-8")
    assert first_spec["source_sha256"] in svg
    assert "95% bootstrap intervals" in svg


def test_jsonl_cli_validates_and_generates(
    synthetic_records: list[dict[str, Any]],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "results.jsonl"
    input_path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in synthetic_records)
        + "\n",
        encoding="utf-8",
    )
    assert main([str(input_path), "--validate-only"]) == 0
    validation_output = capsys.readouterr().out.strip()
    assert len(validation_output) == 64

    output_dir = tmp_path / "report"
    assert (
        main(
            [
                str(input_path),
                "--output-dir",
                str(output_dir),
                "--bootstrap-samples",
                "200",
            ]
        )
        == 0
    )
    assert (output_dir / "report-spec.json").is_file()
