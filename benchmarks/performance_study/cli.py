from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from .core import (
    HarnessError,
    ProbeSnapshot,
    _plan_for_attempt,
    append_result_record,
    build_result_record,
    canonical_hash,
    canonical_json,
    check_preflight,
    collect_live_probe,
    create_manifest,
    derive_run_plan,
    format_plan,
    import_raw_artifacts,
    load_frozen_manifest,
    load_run_events,
    sha256_file,
    validate_raw_run,
    write_frozen_manifest,
)


def _load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"cannot load {label} {path}: {exc}") from exc


def _repo_root(value: str) -> Path:
    return Path(value).resolve()


def _events_default(manifest: dict[str, Any], repo_root: Path) -> Path:
    return repo_root / manifest["artifact_root"] / "events.jsonl"


def _manifest(args: argparse.Namespace) -> tuple[dict[str, Any], str, Path, Path]:
    repo_root = _repo_root(args.repo_root)
    path = Path(args.manifest)
    if not path.is_absolute():
        path = repo_root / path
    manifest, digest = load_frozen_manifest(path)
    return manifest, digest, path, repo_root


def _events_path(
    args: argparse.Namespace, manifest: dict[str, Any], repo_root: Path
) -> Path:
    if args.events is None:
        return _events_default(manifest, repo_root)
    path = Path(args.events)
    return path if path.is_absolute() else repo_root / path


def command_init(args: argparse.Namespace) -> int:
    repo_root = _repo_root(args.repo_root)
    spec = _load_json(Path(args.spec), "study spec")
    manifest = create_manifest(spec, repo_root)
    output = Path(args.output)
    if not output.is_absolute():
        output = repo_root / output
    digest = write_frozen_manifest(manifest, output)
    print(canonical_json({"manifest": output.as_posix(), "manifest_sha256": digest}))
    return 0


def command_validate(args: argparse.Namespace) -> int:
    manifest, digest, path, _ = _manifest(args)
    plan = derive_run_plan(manifest, path, digest)
    blocked = sorted(
        {
            item["scope"]: item["blocked_reason"]
            for item in plan
            if item["adapter_status"] == "blocked"
        }.items()
    )
    print(
        canonical_json(
            {
                "manifest_sha256": digest,
                "runs": len(plan),
                "blocked_adapters": dict(blocked),
            }
        )
    )
    return 0


def command_plan(args: argparse.Namespace) -> int:
    manifest, digest, path, repo_root = _manifest(args)
    events_path = _events_path(args, manifest, repo_root)
    events = load_run_events(events_path, digest)
    plan = derive_run_plan(
        manifest,
        path.relative_to(repo_root) if path.is_relative_to(repo_root) else path,
        digest,
        events,
        args.retry_status,
    )
    if args.format == "json":
        print(json.dumps(plan, sort_keys=True, indent=2))
    else:
        display_manifest = (
            path.relative_to(repo_root) if path.is_relative_to(repo_root) else path
        )
        display_events = (
            events_path.relative_to(repo_root)
            if events_path.is_relative_to(repo_root)
            else events_path
        )
        sys.stdout.write(format_plan(plan, display_manifest, display_events))
    return 0


def command_preflight(args: argparse.Namespace) -> int:
    manifest, _, _, repo_root = _manifest(args)
    if args.probe_fixture:
        snapshot = ProbeSnapshot.from_json(
            _load_json(Path(args.probe_fixture), "probe fixture")
        )
    else:
        snapshot = collect_live_probe(manifest, repo_root)
    gates = check_preflight(manifest, snapshot)
    print(json.dumps(gates, sort_keys=True, indent=2))
    return 0 if all(item["passed"] for item in gates) else 2


def command_import(args: argparse.Namespace) -> int:
    manifest, digest, path, repo_root = _manifest(args)
    planned_manifest_path = (
        path.relative_to(repo_root) if path.is_relative_to(repo_root) else path
    )
    events_path = _events_path(args, manifest, repo_root)
    raw_paths = [
        Path(item) if Path(item).is_absolute() else repo_root / item
        for item in args.raw
    ]
    summary = import_raw_artifacts(
        manifest,
        planned_manifest_path,
        digest,
        events_path,
        raw_paths,
        repo_root,
    )
    summary["result"] = (
        "not-requested" if summary["terminal_coverage"] else "incomplete"
    )
    if summary["terminal_coverage"] and args.results:
        results_path = Path(args.results)
        if not results_path.is_absolute():
            results_path = repo_root / results_path
        events = load_run_events(events_path, digest)
        result = build_result_record(
            manifest,
            planned_manifest_path,
            digest,
            events,
            repo_root,
            args.result_attempt_index,
        )
        summary["result"] = (
            "appended" if append_result_record(results_path, result) else "idempotent"
        )
        summary["result_status"] = result["status"]
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


def _crash_raw(
    plan: dict[str, Any],
    manifest: dict[str, Any],
    digest: str,
    gates: list[dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    from .core import RAW_RUN_RECORD_TYPE, utc_now

    now = utc_now()
    return {
        "schema_version": 1,
        "record_type": RAW_RUN_RECORD_TYPE,
        "manifest_sha256": digest,
        "run_id": plan["run_id"],
        "attempt_id": plan["attempt_id"],
        "attempt_index": plan["attempt_index"],
        "pair_id": plan["pair_id"],
        "sequence_index": plan["sequence_index"],
        "scene_id": plan["scene_id"],
        "scope": plan["scope"],
        "seed": plan["seed"],
        "side": plan["side"],
        "status": "crash",
        "started_at_utc": now,
        "completed_at_utc": now,
        "command_argv": plan["command_argv"],
        "gates": gates,
        "dispatch_proof": {
            "requested_backend": plan["backend"],
            "resolved_backend": "unavailable",
            "artifact_sha256": plan["artifact_sha256"],
            "proof_sha256": canonical_hash({"reason": reason}),
        },
        "timing": {
            "unit": "s" if plan["scope"] == "training" else "ms",
            "samples": [],
            "warmup_iterations": manifest["protocol"]["warmup_iterations"],
            "measured_iterations": manifest["protocol"]["measured_iterations"],
            "rounds": manifest["protocol"]["rounds"],
            "training_steps": manifest["protocol"]["training_steps"],
            "startup_compile_excluded": True,
            "direct_measurement": plan["scope"] == "forward_backward",
        },
        "telemetry": {
            "temperature_start_c": 0.0,
            "temperature_end_c": 0.0,
            "sm_clock_samples_mhz": [],
            "memory_clock_samples_mhz": [],
            "power_samples_w": [],
            "active_gpu_compute_pids_before": [],
            "active_gpu_compute_pids_after": [],
            "throttle_reasons": [],
            "xid_errors": [],
        },
        "correctness": {
            "passed": False,
            "gate": "executor did not complete",
            "artifact_sha256": None,
        },
        "artifact_hashes": {},
        "rejection_reasons": [reason],
    }


def command_execute(args: argparse.Namespace) -> int:
    manifest, digest, path, repo_root = _manifest(args)
    planned_manifest_path = (
        path.relative_to(repo_root) if path.is_relative_to(repo_root) else path
    )
    if os.environ.get("PTXSPLAT_PERFORMANCE_EXCLUSIVE_GPU") != "YES":
        raise HarnessError("execute requires PTXSPLAT_PERFORMANCE_EXCLUSIVE_GPU=YES")
    events_path = _events_path(args, manifest, repo_root)
    events = load_run_events(events_path, digest)
    plan = derive_run_plan(
        manifest,
        planned_manifest_path,
        digest,
        events,
        retry_statuses=args.retry_status,
    )
    matches = [item for item in plan if item["run_id"] == args.run_id]
    if not matches:
        raise HarnessError(f"{args.run_id}: no missing/retryable run")
    item = matches[0]
    if item["adapter_status"] != "ready":
        raise HarnessError(f"{args.run_id}: adapter blocked: {item['blocked_reason']}")
    gates = check_preflight(manifest, collect_live_probe(manifest, repo_root))
    failed = [gate["name"] for gate in gates if not gate["passed"]]
    if failed:
        raise HarnessError("preflight rejected execution: " + ", ".join(failed))

    output_path = Path(item["output_path"])
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    dispatch_path = Path(item["dispatch_path"])
    if not dispatch_path.is_absolute():
        dispatch_path = repo_root / dispatch_path
    if output_path.exists():
        raw = validate_raw_run(
            _load_json(output_path, "raw artifact"), item, manifest, digest
        )
        summary = import_raw_artifacts(
            manifest,
            planned_manifest_path,
            digest,
            events_path,
            [output_path],
            repo_root,
        )
        print(json.dumps(summary, sort_keys=True, indent=2))
        return 0 if raw["status"] == "complete" else 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(item["dispatch_argv"], cwd=repo_root, check=True)
        dispatch = _load_json(dispatch_path, "dispatch proof")
        expected_dispatch = {
            "requested_backend": manifest["variants"][item["side"]]["backend"],
            "resolved_backend": manifest["variants"][item["side"]]["expected_dispatch"],
            "artifact_sha256": item["artifact_sha256"],
        }
        if dispatch != expected_dispatch:
            raise HarnessError(f"dispatch proof mismatch: {dispatch!r}")
        subprocess.run(item["command_argv"], cwd=repo_root, check=True)
        raw = _load_json(output_path, "raw artifact")
        required_gates = {gate["name"]: gate for gate in gates}
        raw_gates = {gate["name"]: gate for gate in raw.get("gates", [])}
        if any(raw_gates.get(name) != gate for name, gate in required_gates.items()):
            raise HarnessError("raw artifact does not preserve exact preflight gates")
        expected_proof = {
            **expected_dispatch,
            "proof_sha256": sha256_file(dispatch_path),
        }
        if raw.get("dispatch_proof") != expected_proof:
            raise HarnessError("raw artifact dispatch proof/hash mismatch")
        validate_raw_run(raw, item, manifest, digest)
    except (OSError, subprocess.CalledProcessError, HarnessError) as exc:
        crash = _crash_raw(item, manifest, digest, gates, str(exc))
        output_path.write_text(canonical_json(crash) + "\n", encoding="ascii")
    summary = import_raw_artifacts(
        manifest,
        planned_manifest_path,
        digest,
        events_path,
        [output_path],
        repo_root,
    )
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0 if _load_json(output_path, "raw artifact")["status"] == "complete" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPU-deferred ptxsplat performance study harness"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="freeze a JSON study spec without CUDA")
    init.add_argument("--spec", required=True)
    init.add_argument("--output", required=True)
    init.add_argument("--repo-root", default=".")
    init.set_defaults(handler=command_init)

    for name in ("validate", "plan", "preflight", "import", "execute"):
        child = subparsers.add_parser(name)
        child.add_argument("--manifest", required=True)
        child.add_argument("--repo-root", default=".")
        if name in ("plan", "import", "execute"):
            child.add_argument("--events")
        if name == "validate":
            child.set_defaults(handler=command_validate)
        elif name == "plan":
            child.add_argument("--format", choices=("text", "json"), default="text")
            child.add_argument("--retry-status", action="append", default=[])
            child.set_defaults(handler=command_plan)
        elif name == "preflight":
            child.add_argument("--probe-fixture")
            child.set_defaults(handler=command_preflight)
        elif name == "import":
            child.add_argument("--raw", action="append", required=True)
            child.add_argument("--results")
            child.add_argument("--result-attempt-index", type=int, default=0)
            child.set_defaults(handler=command_import)
        else:
            child.add_argument("--run-id", required=True)
            child.add_argument("--retry-status", action="append", default=[])
            child.set_defaults(handler=command_execute)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except HarnessError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
