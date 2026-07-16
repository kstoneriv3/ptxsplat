from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from benchmarks.performance_history.core import validate_record
from benchmarks.performance_study.core import (
    HarnessError,
    ProbeSnapshot,
    append_run_event,
    build_result_record,
    canonical_json,
    check_preflight,
    create_manifest,
    derive_run_plan,
    import_raw_artifacts,
    load_frozen_manifest,
    load_run_events,
    validate_manifest,
    validate_raw_run,
    write_frozen_manifest,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _adapter(status: str = "ready", reason: str = "") -> dict[str, Any]:
    if status == "blocked":
        return {
            "status": "blocked",
            "adapter": "deferred_executor",
            "reason": reason,
            "command_argv": None,
            "dispatch_argv": None,
            "startup_compile_contract": "compile preflight then reset before timing",
        }
    return {
        "status": "ready",
        "adapter": "fixture_executor",
        "reason": "",
        "command_argv": [
            "python3",
            "fixture_executor.py",
            "--output",
            "{output_path}",
            "--scope",
            "{scope}",
            "--seed",
            "{seed}",
        ],
        "dispatch_argv": [
            "python3",
            "fixture_dispatch.py",
            "--output",
            "{dispatch_path}",
            "--backend",
            "{backend}",
        ],
        "startup_compile_contract": (
            "compile preflight, reset state, and directly measure the requested scope"
        ),
    }


@pytest.fixture
def frozen_study(tmp_path: Path) -> tuple[Path, dict[str, Any], Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "study@example.invalid")
    _git(repo, "config", "user.name", "Study Test")
    (repo / ".gitignore").write_text("benchmark-results/\n", encoding="ascii")
    (repo / "evaluator.py").write_text("EVALUATOR_VERSION = 1\n", encoding="ascii")
    (repo / "artifact.bin").write_bytes(b"frozen-extension")
    dataset = repo / "data" / "bonsai"
    dataset.mkdir(parents=True)
    (dataset / "image.bin").write_bytes(b"bonsai")
    (dataset / "manifest.sha256").write_text(
        "placeholder inventory pinned by its outer hash\n", encoding="ascii"
    )
    _git(repo, "add", ".gitignore", "evaluator.py", "artifact.bin", "data")
    _git(repo, "commit", "-qm", "fixture")

    environment = {
        "environment_id": "fixture-rtx5090",
        "container_image": "360-video-gs-dev:latest",
        "container_image_id": "sha256:" + "a" * 64,
        "gpu_index": 0,
        "gpu_name": "NVIDIA GeForce RTX 5090",
        "gpu_uuid": "GPU-fixture",
        "driver_version": "fixture-driver",
        "cuda_version": "12.8",
        "pytorch_version": "2.9.1",
        "build_flags": ["-O3", "--use_fast_math", "sm_120"],
        "sm_clock_mhz": 2400,
        "memory_clock_mhz": 14000,
        "viewer_containers": [
            "ptxsplat-ns-compare-upstream",
            "ptxsplat-ns-compare-sm120",
        ],
    }
    spec = {
        "study_id": "fixture-study",
        "experiment_id": "current-vs-reference",
        "description": "Synthetic harness validation only",
        "comparison_kind": "candidate",
        "artifact_root": "benchmark-results/performance-study/fixture-study",
        "repository": {
            "evaluator_revision": "HEAD",
            "evaluator_paths": [{"name": "evaluator", "path": "evaluator.py"}],
        },
        "environment": environment,
        "variants": {
            "baseline": {
                "label": "Forced reference",
                "revision": "HEAD",
                "backend": "reference",
                "expected_dispatch": "reference",
                "artifact": "extension",
                "environment": {"PTXSPLAT_BACKEND": "reference"},
            },
            "candidate": {
                "label": "SM120 candidate",
                "revision": "HEAD",
                "backend": "sm120",
                "expected_dispatch": "sm120",
                "artifact": "extension",
                "environment": {"PTXSPLAT_BACKEND": "sm120"},
            },
        },
        "datasets": [
            {
                "scene_id": "bonsai",
                "dataset": "mipnerf360",
                "path": "data/bonsai",
                "manifest_path": "data/bonsai/manifest.sha256",
                "data_factor": 2,
            }
        ],
        "protocol": {
            "seeds": [42, 43, 44],
            "pair_order": [
                "baseline-first",
                "candidate-first",
                "baseline-first",
            ],
            "scopes": [
                "training",
                "isolated_forward",
                "isolated_backward",
                "forward_backward",
            ],
            "training_steps": 7000,
            "warmup_iterations": 20,
            "measured_iterations": 100,
            "rounds": 2,
            "startup_compile_excluded": True,
            "startup_compile_contract": (
                "load and compile sacrificially, reset all state, then start timing"
            ),
            "progress_scene_id": "bonsai",
            "progress_scope": "forward_backward",
            "publication": False,
        },
        "adapters": {
            scope: _adapter()
            for scope in (
                "training",
                "isolated_forward",
                "isolated_backward",
                "forward_backward",
            )
        },
        "frozen_artifacts": [{"name": "extension", "path": "artifact.bin"}],
    }
    manifest = create_manifest(spec, repo)
    manifest_path = repo / "benchmark-results" / "performance-study" / "manifest.json"
    digest = write_frozen_manifest(manifest, manifest_path)
    return repo, manifest, manifest_path, digest


def _raw_run(
    item: dict[str, Any],
    manifest: dict[str, Any],
    digest: str,
    *,
    status: str = "complete",
) -> dict[str, Any]:
    seconds = item["sequence_index"] % 60
    samples = [10.0 + item["repeat_index"]]
    if item["scope"] != "training":
        samples = [1.0 + item["repeat_index"], 1.1 + item["repeat_index"]]
    side_offset = 0.0 if item["side"] == "baseline" else -0.1
    samples = [value + side_offset for value in samples]
    variant = manifest["variants"][item["side"]]
    raw = {
        "schema_version": 1,
        "record_type": "ptxsplat.performance_study.raw_run",
        "manifest_sha256": digest,
        "run_id": item["run_id"],
        "attempt_id": item["attempt_id"],
        "attempt_index": item["attempt_index"],
        "pair_id": item["pair_id"],
        "sequence_index": item["sequence_index"],
        "scene_id": item["scene_id"],
        "scope": item["scope"],
        "seed": item["seed"],
        "side": item["side"],
        "status": status,
        "started_at_utc": f"2026-07-16T12:00:{seconds:02d}Z",
        "completed_at_utc": f"2026-07-16T12:01:{seconds:02d}Z",
        "command_argv": item["command_argv"],
        "gates": [
            {
                "name": "fixture_preflight",
                "passed": status == "complete",
                "evidence": "fixture",
            }
        ],
        "dispatch_proof": {
            "requested_backend": variant["backend"],
            "resolved_backend": variant["expected_dispatch"],
            "artifact_sha256": item["artifact_sha256"],
            "proof_sha256": "d" * 64,
        },
        "timing": {
            "unit": "s" if item["scope"] == "training" else "ms",
            "samples": samples,
            "warmup_iterations": manifest["protocol"]["warmup_iterations"],
            "measured_iterations": manifest["protocol"]["measured_iterations"],
            "rounds": manifest["protocol"]["rounds"],
            "training_steps": manifest["protocol"]["training_steps"],
            "startup_compile_excluded": True,
            "direct_measurement": item["scope"] == "forward_backward",
        },
        "telemetry": {
            "temperature_start_c": 62.0,
            "temperature_end_c": 65.0,
            "sm_clock_samples_mhz": [2400.0, 2400.0],
            "memory_clock_samples_mhz": [14000.0, 14000.0],
            "power_samples_w": [300.0, 310.0],
            "active_gpu_compute_pids_before": [],
            "active_gpu_compute_pids_after": [],
            "throttle_reasons": [],
            "xid_errors": [],
        },
        "correctness": {
            "passed": status == "complete",
            "gate": "fixture parity",
            "artifact_sha256": "c" * 64 if status == "complete" else None,
        },
        "artifact_hashes": {"fixture": "e" * 64},
        "rejection_reasons": [] if status == "complete" else ["fixture rejection"],
    }
    return raw


def _passing_probe(manifest: dict[str, Any]) -> ProbeSnapshot:
    return ProbeSnapshot(
        viewer_states={
            name: "exited" for name in manifest["environment"]["viewer_containers"]
        },
        gpu_compute_pids=(),
        tracked_tree_clean=True,
        head_commit=manifest["repository"]["evaluator_commit"],
        tracked_tree_sha256=manifest["repository"]["tracked_tree_sha256"],
        container_image_id=manifest["environment"]["container_image_id"],
        dataset_manifest_sha256={
            item["scene_id"]: item["manifest_sha256"] for item in manifest["datasets"]
        },
        artifact_sha256={
            item["name"]: item["sha256"] for item in manifest["frozen_artifacts"]
        },
    )


def test_frozen_manifest_detects_hash_drift(
    frozen_study: tuple[Path, dict[str, Any], Path, str]
) -> None:
    _, manifest, path, digest = frozen_study
    loaded, loaded_digest = load_frozen_manifest(path)
    assert loaded == manifest
    assert loaded_digest == digest

    changed = copy.deepcopy(manifest)
    changed["description"] = "mutated"
    path.write_text(canonical_json(changed) + "\n", encoding="ascii")
    with pytest.raises(HarnessError, match="hash drift"):
        load_frozen_manifest(path)


def test_plan_is_deterministic_balanced_and_combined_scope_is_distinct(
    frozen_study: tuple[Path, dict[str, Any], Path, str]
) -> None:
    _, manifest, path, digest = frozen_study
    plan = derive_run_plan(manifest, path, digest)
    assert len(plan) == 24
    assert plan == derive_run_plan(manifest, path, digest)
    training = [item["side"] for item in plan if item["scope"] == "training"]
    assert training == [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
        "baseline",
        "candidate",
    ]
    combined = [item for item in plan if item["scope"] == "forward_backward"]
    assert len(combined) == 6
    assert all(item["scope"] == "forward_backward" for item in combined)


def test_blocked_adapter_never_advertises_command(
    frozen_study: tuple[Path, dict[str, Any], Path, str]
) -> None:
    _, manifest, path, digest = frozen_study
    blocked = copy.deepcopy(manifest)
    blocked["adapters"]["training"] = _adapter(
        "blocked", "trainer lacks resettable startup-excluded timing"
    )
    validate_manifest(blocked)
    plan = derive_run_plan(blocked, path, digest)
    training = [item for item in plan if item["scope"] == "training"]
    assert all(item["adapter_status"] == "blocked" for item in training)
    assert all(item["command_argv"] is None for item in training)


def test_resume_and_retry_preserve_logical_sequence(
    frozen_study: tuple[Path, dict[str, Any], Path, str]
) -> None:
    repo, manifest, path, digest = frozen_study
    first = derive_run_plan(manifest, path, digest)[0]
    raw = _raw_run(first, manifest, digest, status="rejected")
    raw_path = repo / first["output_path"]
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text(canonical_json(raw) + "\n", encoding="ascii")
    events_path = repo / manifest["artifact_root"] / "events.jsonl"
    import_raw_artifacts(manifest, path, digest, events_path, [raw_path], repo)
    events = load_run_events(events_path, digest)

    assert len(derive_run_plan(manifest, path, digest, events)) == 23
    retry = derive_run_plan(
        manifest, path, digest, events, retry_statuses=("rejected",)
    )
    retried = next(item for item in retry if item["run_id"] == first["run_id"])
    assert retried["attempt_index"] == 1
    assert retried["sequence_index"] == first["sequence_index"]


def test_preflight_rejects_viewer_and_gpu_contamination(
    frozen_study: tuple[Path, dict[str, Any], Path, str]
) -> None:
    _, manifest, _, _ = frozen_study
    passing = _passing_probe(manifest)
    assert all(item["passed"] for item in check_preflight(manifest, passing))

    contaminated = ProbeSnapshot(
        **{
            **passing.__dict__,
            "viewer_states": {
                **passing.viewer_states,
                "ptxsplat-ns-compare-upstream": "running",
            },
            "gpu_compute_pids": (1234,),
        }
    )
    failed = {
        item["name"]
        for item in check_preflight(manifest, contaminated)
        if not item["passed"]
    }
    assert failed == {"viewer_containers_stopped", "gpu_compute_exclusive"}


def test_raw_validation_rejects_manifest_dispatch_and_direct_timing_drift(
    frozen_study: tuple[Path, dict[str, Any], Path, str]
) -> None:
    _, manifest, path, digest = frozen_study
    item = next(
        item
        for item in derive_run_plan(manifest, path, digest)
        if item["scope"] == "forward_backward" and item["side"] == "candidate"
    )
    raw = _raw_run(item, manifest, digest)
    validate_raw_run(raw, item, manifest, digest)

    changed = copy.deepcopy(raw)
    changed["timing"]["direct_measurement"] = False
    with pytest.raises(HarnessError, match="directly measured"):
        validate_raw_run(changed, item, manifest, digest)
    changed = copy.deepcopy(raw)
    changed["manifest_sha256"] = "0" * 64
    with pytest.raises(HarnessError, match="manifest hash drift"):
        validate_raw_run(changed, item, manifest, digest)
    changed = copy.deepcopy(raw)
    changed["dispatch_proof"]["resolved_backend"] = "reference"
    with pytest.raises(HarnessError, match="fallback"):
        validate_raw_run(changed, item, manifest, digest)


def test_duplicate_event_id_is_idempotent_only_for_identical_content(
    frozen_study: tuple[Path, dict[str, Any], Path, str]
) -> None:
    repo, manifest, path, digest = frozen_study
    item = derive_run_plan(manifest, path, digest)[0]
    raw = _raw_run(item, manifest, digest)
    events = repo / manifest["artifact_root"] / "events.jsonl"
    artifact = Path(item["output_path"])
    assert append_run_event(events, raw, artifact)[1]
    assert not append_run_event(events, raw, artifact)[1]
    changed = copy.deepcopy(raw)
    changed["timing"]["samples"][0] += 1
    with pytest.raises(HarnessError, match="different content"):
        append_run_event(events, changed, artifact)


def test_complete_import_builds_schema_valid_result_and_incomplete_is_explicit(
    frozen_study: tuple[Path, dict[str, Any], Path, str]
) -> None:
    repo, manifest, path, digest = frozen_study
    events_path = repo / manifest["artifact_root"] / "events.jsonl"
    plan = derive_run_plan(manifest, path, digest)
    raw_paths: list[Path] = []
    for item in plan:
        raw_path = repo / item["output_path"]
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(
            canonical_json(_raw_run(item, manifest, digest)) + "\n",
            encoding="ascii",
        )
        raw_paths.append(raw_path)

    partial = import_raw_artifacts(
        manifest, path, digest, events_path, raw_paths[:1], repo
    )
    assert not partial["terminal_coverage"]
    assert partial["coverage"]["missing"]

    complete = import_raw_artifacts(
        manifest, path, digest, events_path, raw_paths[1:], repo
    )
    assert complete["terminal_coverage"]
    events = load_run_events(events_path, digest)
    record = build_result_record(manifest, path, digest, events, repo, 0)
    assert validate_record(record) == record
    assert record["status"] == "keep"
    assert len(record["measurements"]) == 4
