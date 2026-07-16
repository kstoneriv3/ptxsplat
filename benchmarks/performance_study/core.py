from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from benchmarks.performance_history.core import validate_record

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_RECORD_TYPE = "ptxsplat.performance_study.manifest"
RAW_RUN_RECORD_TYPE = "ptxsplat.performance_study.raw_run"
RUN_EVENT_RECORD_TYPE = "ptxsplat.performance_study.run_event"
SCOPES = (
    "training",
    "isolated_forward",
    "isolated_backward",
    "forward_backward",
)
SIDES = ("baseline", "candidate")
TERMINAL_STATUSES = ("complete", "rejected", "crash")
ADAPTER_STATUSES = ("ready", "blocked")
PAIR_ORDERS = ("baseline-first", "candidate-first")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")

TEMPLATE_KEYS = frozenset(
    {
        "adapter",
        "artifact_root",
        "attempt_id",
        "attempt_index",
        "backend",
        "checkpoint_path",
        "data_factor",
        "dataset_path",
        "dispatch_path",
        "manifest_path",
        "manifest_sha256",
        "measured_iterations",
        "output_path",
        "pair_id",
        "rounds",
        "run_id",
        "scene_id",
        "scope",
        "seed",
        "side",
        "training_steps",
        "variant",
        "warmup_iterations",
    }
)


class HarnessError(ValueError):
    """Raised when study provenance or execution state is not trustworthy."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise HarnessError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("ascii"))


def _require(condition: bool, path: str, message: str) -> None:
    if not condition:
        raise HarnessError(f"{path}: {message}")


def _dict(value: Any, path: str) -> dict[str, Any]:
    _require(isinstance(value, dict), path, "expected an object")
    return value


def _list(value: Any, path: str) -> list[Any]:
    _require(isinstance(value, list), path, "expected an array")
    return value


def _str(value: Any, path: str) -> str:
    _require(isinstance(value, str) and bool(value), path, "expected a string")
    return value


def _int(value: Any, path: str, minimum: int = 0) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        path,
        f"expected an integer >= {minimum}",
    )
    return value


def _float(value: Any, path: str, *, positive: bool = False) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        path,
        "expected a number",
    )
    result = float(value)
    _require(math.isfinite(result), path, "expected a finite number")
    if positive:
        _require(result > 0, path, "expected a positive number")
    return result


def _exact_keys(
    value: Mapping[str, Any],
    required: Iterable[str],
    optional: Iterable[str],
    path: str,
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - set(value))
    extra = sorted(set(value) - allowed)
    _require(not missing, path, f"missing fields: {', '.join(missing)}")
    _require(not extra, path, f"unknown fields: {', '.join(extra)}")


def _id(value: Any, path: str) -> str:
    result = _str(value, path)
    _require(bool(_ID.fullmatch(result)), path, "invalid identifier")
    return result


def _hex(value: Any, path: str, length: int) -> str:
    result = _str(value, path)
    pattern = _HEX40 if length == 40 else _HEX64
    _require(bool(pattern.fullmatch(result)), path, f"expected {length} lowercase hex")
    return result


def _relative(value: Any, path: str) -> str:
    result = _str(value, path)
    candidate = Path(result)
    _require(not candidate.is_absolute(), path, "must be relative to repository root")
    _require(".." not in candidate.parts, path, "must not escape repository root")
    return candidate.as_posix()


def _argv(value: Any, path: str) -> list[str]:
    result = _list(value, path)
    _require(bool(result), path, "argv must not be empty")
    for index, item in enumerate(result):
        token = _str(item, f"{path}[{index}]")
        unknown = sorted(set(_PLACEHOLDER.findall(token)) - TEMPLATE_KEYS)
        _require(
            not unknown,
            f"{path}[{index}]",
            f"unknown placeholders: {', '.join(unknown)}",
        )
    return result


def _hash_path_entries(
    entries: Sequence[Mapping[str, Any]], repo_root: Path, path: str
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    names: set[str] = set()
    for index, raw in enumerate(entries):
        item = _dict(raw, f"{path}[{index}]")
        _exact_keys(item, ("name", "path"), (), f"{path}[{index}]")
        name = _id(item["name"], f"{path}[{index}].name")
        _require(name not in names, f"{path}[{index}].name", "duplicate name")
        names.add(name)
        relative = _relative(item["path"], f"{path}[{index}].path")
        full_path = repo_root / relative
        _require(full_path.is_file(), str(full_path), "artifact does not exist")
        result.append(
            {"name": name, "path": relative, "sha256": sha256_file(full_path)}
        )
    return result


def _run_git(repo_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise HarnessError(f"git {' '.join(args)} failed: {detail.strip()}") from exc
    return completed.stdout.strip()


def _resolve_commit(repo_root: Path, revision: str) -> str:
    commit = _run_git(repo_root, "rev-parse", f"{revision}^{{commit}}")
    return _hex(commit, "commit", 40)


def _tracked_tree_hash(repo_root: Path, commit: str) -> str:
    listing = _run_git(repo_root, "ls-tree", "-r", "--full-tree", commit)
    return sha256_bytes((listing + "\n").encode("utf-8"))


def create_manifest(spec: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    """Resolve a human-authored spec into an immutable, hashable manifest."""

    source = _dict(dict(spec), "spec")
    _exact_keys(
        source,
        (
            "study_id",
            "experiment_id",
            "artifact_root",
            "repository",
            "environment",
            "variants",
            "datasets",
            "protocol",
            "adapters",
            "frozen_artifacts",
        ),
        ("description", "comparison_kind"),
        "spec",
    )
    repository = _dict(source["repository"], "spec.repository")
    _exact_keys(
        repository, ("evaluator_revision", "evaluator_paths"), (), "spec.repository"
    )
    evaluator_commit = _resolve_commit(
        repo_root,
        _str(repository["evaluator_revision"], "spec.repository.evaluator_revision"),
    )
    evaluator_entries = _hash_path_entries(
        _list(repository["evaluator_paths"], "spec.repository.evaluator_paths"),
        repo_root,
        "spec.repository.evaluator_paths",
    )
    _require(
        bool(evaluator_entries), "spec.repository.evaluator_paths", "must not be empty"
    )
    artifacts = _hash_path_entries(
        _list(source["frozen_artifacts"], "spec.frozen_artifacts"),
        repo_root,
        "spec.frozen_artifacts",
    )

    variants_source = _dict(source["variants"], "spec.variants")
    _exact_keys(variants_source, SIDES, (), "spec.variants")
    variants: dict[str, Any] = {}
    artifact_names = {item["name"] for item in artifacts}
    for side in SIDES:
        item = _dict(variants_source[side], f"spec.variants.{side}")
        _exact_keys(
            item,
            ("label", "revision", "backend", "expected_dispatch", "artifact"),
            ("environment",),
            f"spec.variants.{side}",
        )
        artifact = _id(item["artifact"], f"spec.variants.{side}.artifact")
        _require(
            artifact in artifact_names,
            f"spec.variants.{side}.artifact",
            "unknown artifact",
        )
        environment = _dict(
            item.get("environment", {}), f"spec.variants.{side}.environment"
        )
        for key, value in environment.items():
            _require(
                isinstance(key, str) and key and isinstance(value, str),
                f"spec.variants.{side}.environment",
                "environment keys and values must be strings",
            )
        variants[side] = {
            "label": _str(item["label"], f"spec.variants.{side}.label"),
            "commit": _resolve_commit(
                repo_root, _str(item["revision"], f"spec.variants.{side}.revision")
            ),
            "backend": _str(item["backend"], f"spec.variants.{side}.backend"),
            "expected_dispatch": _str(
                item["expected_dispatch"], f"spec.variants.{side}.expected_dispatch"
            ),
            "artifact": artifact,
            "environment": dict(sorted(environment.items())),
        }

    datasets: list[dict[str, Any]] = []
    scene_ids: set[str] = set()
    for index, raw in enumerate(_list(source["datasets"], "spec.datasets")):
        item = _dict(raw, f"spec.datasets[{index}]")
        _exact_keys(
            item,
            ("scene_id", "dataset", "path", "manifest_path", "data_factor"),
            (),
            f"spec.datasets[{index}]",
        )
        scene = _id(item["scene_id"], f"spec.datasets[{index}].scene_id")
        _require(
            scene not in scene_ids,
            f"spec.datasets[{index}].scene_id",
            "duplicate scene",
        )
        scene_ids.add(scene)
        dataset_path = _relative(item["path"], f"spec.datasets[{index}].path")
        manifest_path = _relative(
            item["manifest_path"], f"spec.datasets[{index}].manifest_path"
        )
        _require(
            (repo_root / dataset_path).is_dir(),
            dataset_path,
            "dataset directory is absent",
        )
        _require(
            (repo_root / manifest_path).is_file(),
            manifest_path,
            "dataset manifest is absent",
        )
        datasets.append(
            {
                "scene_id": scene,
                "dataset": _str(item["dataset"], f"spec.datasets[{index}].dataset"),
                "path": dataset_path,
                "manifest_path": manifest_path,
                "manifest_sha256": sha256_file(repo_root / manifest_path),
                "data_factor": _int(
                    item["data_factor"], f"spec.datasets[{index}].data_factor", 1
                ),
            }
        )

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "record_type": MANIFEST_RECORD_TYPE,
        "study_id": _id(source["study_id"], "spec.study_id"),
        "experiment_id": _id(source["experiment_id"], "spec.experiment_id"),
        "description": _str(
            source.get("description", "Deferred performance study"), "spec.description"
        ),
        "comparison_kind": source.get("comparison_kind", "candidate"),
        "artifact_root": _relative(source["artifact_root"], "spec.artifact_root"),
        "repository": {
            "evaluator_commit": evaluator_commit,
            "tracked_tree_sha256": _tracked_tree_hash(repo_root, evaluator_commit),
            "evaluator_paths": evaluator_entries,
            "evaluator_sha256": canonical_hash(evaluator_entries),
        },
        "environment": source["environment"],
        "variants": variants,
        "datasets": datasets,
        "protocol": source["protocol"],
        "adapters": source["adapters"],
        "frozen_artifacts": artifacts,
    }
    return validate_manifest(manifest)


def validate_manifest(value: Any) -> dict[str, Any]:
    manifest = _dict(value, "manifest")
    required = (
        "schema_version",
        "record_type",
        "study_id",
        "experiment_id",
        "description",
        "comparison_kind",
        "artifact_root",
        "repository",
        "environment",
        "variants",
        "datasets",
        "protocol",
        "adapters",
        "frozen_artifacts",
    )
    _exact_keys(manifest, required, (), "manifest")
    _require(manifest["schema_version"] == 1, "manifest.schema_version", "expected 1")
    _require(
        manifest["record_type"] == MANIFEST_RECORD_TYPE,
        "manifest.record_type",
        "invalid type",
    )
    _id(manifest["study_id"], "manifest.study_id")
    _id(manifest["experiment_id"], "manifest.experiment_id")
    _str(manifest["description"], "manifest.description")
    _require(
        manifest["comparison_kind"]
        in ("baseline", "historical", "ablation", "candidate"),
        "manifest.comparison_kind",
        "invalid comparison kind",
    )
    _relative(manifest["artifact_root"], "manifest.artifact_root")

    repository = _dict(manifest["repository"], "manifest.repository")
    _exact_keys(
        repository,
        (
            "evaluator_commit",
            "tracked_tree_sha256",
            "evaluator_paths",
            "evaluator_sha256",
        ),
        (),
        "manifest.repository",
    )
    _hex(repository["evaluator_commit"], "manifest.repository.evaluator_commit", 40)
    _hex(
        repository["tracked_tree_sha256"], "manifest.repository.tracked_tree_sha256", 64
    )
    evaluator_paths = _validate_hashed_paths(
        repository["evaluator_paths"], "manifest.repository.evaluator_paths"
    )
    _require(
        canonical_hash(evaluator_paths) == repository["evaluator_sha256"],
        "manifest.repository.evaluator_sha256",
        "does not match evaluator path records",
    )

    environment = _dict(manifest["environment"], "manifest.environment")
    _exact_keys(
        environment,
        (
            "environment_id",
            "container_image",
            "container_image_id",
            "gpu_index",
            "gpu_name",
            "gpu_uuid",
            "driver_version",
            "cuda_version",
            "pytorch_version",
            "build_flags",
            "sm_clock_mhz",
            "memory_clock_mhz",
            "viewer_containers",
        ),
        (),
        "manifest.environment",
    )
    _id(environment["environment_id"], "manifest.environment.environment_id")
    for key in (
        "container_image",
        "gpu_name",
        "gpu_uuid",
        "driver_version",
        "cuda_version",
        "pytorch_version",
    ):
        _str(environment[key], f"manifest.environment.{key}")
    image_id = _str(
        environment["container_image_id"], "manifest.environment.container_image_id"
    )
    _require(
        image_id.startswith("sha256:") and bool(_HEX64.fullmatch(image_id[7:])),
        "manifest.environment.container_image_id",
        "expected sha256:<64 hex>",
    )
    _int(environment["gpu_index"], "manifest.environment.gpu_index")
    flags = _list(environment["build_flags"], "manifest.environment.build_flags")
    _require(bool(flags), "manifest.environment.build_flags", "must not be empty")
    for index, flag in enumerate(flags):
        _str(flag, f"manifest.environment.build_flags[{index}]")
    _float(
        environment["sm_clock_mhz"], "manifest.environment.sm_clock_mhz", positive=True
    )
    _float(
        environment["memory_clock_mhz"],
        "manifest.environment.memory_clock_mhz",
        positive=True,
    )
    viewers = _list(
        environment["viewer_containers"], "manifest.environment.viewer_containers"
    )
    _require(
        bool(viewers),
        "manifest.environment.viewer_containers",
        "must name viewer containers",
    )
    for index, viewer in enumerate(viewers):
        _str(viewer, f"manifest.environment.viewer_containers[{index}]")
    _require(
        len(viewers) == len(set(viewers)),
        "manifest.environment.viewer_containers",
        "duplicates",
    )

    artifacts = _validate_hashed_paths(
        manifest["frozen_artifacts"], "manifest.frozen_artifacts"
    )
    artifact_names = {item["name"] for item in artifacts}
    variants = _dict(manifest["variants"], "manifest.variants")
    _exact_keys(variants, SIDES, (), "manifest.variants")
    for side in SIDES:
        item = _dict(variants[side], f"manifest.variants.{side}")
        _exact_keys(
            item,
            (
                "label",
                "commit",
                "backend",
                "expected_dispatch",
                "artifact",
                "environment",
            ),
            (),
            f"manifest.variants.{side}",
        )
        _str(item["label"], f"manifest.variants.{side}.label")
        _hex(item["commit"], f"manifest.variants.{side}.commit", 40)
        _str(item["backend"], f"manifest.variants.{side}.backend")
        _str(item["expected_dispatch"], f"manifest.variants.{side}.expected_dispatch")
        _require(
            item["artifact"] in artifact_names,
            f"manifest.variants.{side}.artifact",
            "unknown artifact",
        )
        env = _dict(item["environment"], f"manifest.variants.{side}.environment")
        for key, env_value in env.items():
            _str(key, f"manifest.variants.{side}.environment key")
            _require(
                isinstance(env_value, str),
                f"manifest.variants.{side}.environment.{key}",
                "expected a string",
            )

    datasets = _list(manifest["datasets"], "manifest.datasets")
    _require(bool(datasets), "manifest.datasets", "must not be empty")
    scenes: set[str] = set()
    for index, raw in enumerate(datasets):
        item = _dict(raw, f"manifest.datasets[{index}]")
        _exact_keys(
            item,
            (
                "scene_id",
                "dataset",
                "path",
                "manifest_path",
                "manifest_sha256",
                "data_factor",
            ),
            (),
            f"manifest.datasets[{index}]",
        )
        scene = _id(item["scene_id"], f"manifest.datasets[{index}].scene_id")
        _require(
            scene not in scenes, f"manifest.datasets[{index}].scene_id", "duplicate"
        )
        scenes.add(scene)
        _str(item["dataset"], f"manifest.datasets[{index}].dataset")
        _relative(item["path"], f"manifest.datasets[{index}].path")
        _relative(item["manifest_path"], f"manifest.datasets[{index}].manifest_path")
        _hex(item["manifest_sha256"], f"manifest.datasets[{index}].manifest_sha256", 64)
        _int(item["data_factor"], f"manifest.datasets[{index}].data_factor", 1)

    protocol = _dict(manifest["protocol"], "manifest.protocol")
    _exact_keys(
        protocol,
        (
            "seeds",
            "pair_order",
            "scopes",
            "training_steps",
            "warmup_iterations",
            "measured_iterations",
            "rounds",
            "startup_compile_excluded",
            "startup_compile_contract",
            "progress_scene_id",
            "progress_scope",
            "publication",
        ),
        (),
        "manifest.protocol",
    )
    seeds = _list(protocol["seeds"], "manifest.protocol.seeds")
    _require(
        len(seeds) >= 3,
        "manifest.protocol.seeds",
        "requires at least three paired repeats",
    )
    for index, seed in enumerate(seeds):
        _int(seed, f"manifest.protocol.seeds[{index}]")
    _require(len(seeds) == len(set(seeds)), "manifest.protocol.seeds", "duplicates")
    orders = _list(protocol["pair_order"], "manifest.protocol.pair_order")
    _require(
        len(orders) == len(seeds),
        "manifest.protocol.pair_order",
        "must align with seeds",
    )
    for index, order in enumerate(orders):
        _require(
            order in PAIR_ORDERS,
            f"manifest.protocol.pair_order[{index}]",
            "invalid order",
        )
    _require(
        abs(orders.count(PAIR_ORDERS[0]) - orders.count(PAIR_ORDERS[1])) <= 1,
        "manifest.protocol.pair_order",
        "must be balanced",
    )
    scopes = _list(protocol["scopes"], "manifest.protocol.scopes")
    _require(
        scopes == list(SCOPES), "manifest.protocol.scopes", f"expected {list(SCOPES)!r}"
    )
    _int(protocol["training_steps"], "manifest.protocol.training_steps", 1)
    _int(protocol["warmup_iterations"], "manifest.protocol.warmup_iterations")
    _int(protocol["measured_iterations"], "manifest.protocol.measured_iterations", 1)
    _int(protocol["rounds"], "manifest.protocol.rounds", 1)
    _require(
        protocol["startup_compile_excluded"] is True,
        "manifest.protocol.startup_compile_excluded",
        "must be true",
    )
    _str(
        protocol["startup_compile_contract"],
        "manifest.protocol.startup_compile_contract",
    )
    _require(
        protocol["progress_scene_id"] in scenes,
        "manifest.protocol.progress_scene_id",
        "unknown scene",
    )
    _require(
        protocol["progress_scope"] in SCOPES,
        "manifest.protocol.progress_scope",
        "unknown scope",
    )
    _require(
        isinstance(protocol["publication"], bool),
        "manifest.protocol.publication",
        "expected boolean",
    )

    adapters = _dict(manifest["adapters"], "manifest.adapters")
    _exact_keys(adapters, SCOPES, (), "manifest.adapters")
    for scope in SCOPES:
        _validate_adapter(adapters[scope], scope)
    return manifest


def _validate_hashed_paths(value: Any, path: str) -> list[dict[str, str]]:
    rows = _list(value, path)
    _require(bool(rows), path, "must not be empty")
    names: set[str] = set()
    for index, raw in enumerate(rows):
        item = _dict(raw, f"{path}[{index}]")
        _exact_keys(item, ("name", "path", "sha256"), (), f"{path}[{index}]")
        name = _id(item["name"], f"{path}[{index}].name")
        _require(name not in names, f"{path}[{index}].name", "duplicate")
        names.add(name)
        _relative(item["path"], f"{path}[{index}].path")
        _hex(item["sha256"], f"{path}[{index}].sha256", 64)
    return rows


def _validate_adapter(value: Any, scope: str) -> dict[str, Any]:
    path = f"manifest.adapters.{scope}"
    adapter = _dict(value, path)
    _exact_keys(
        adapter,
        (
            "status",
            "adapter",
            "reason",
            "command_argv",
            "dispatch_argv",
            "startup_compile_contract",
        ),
        (),
        path,
    )
    _require(adapter["status"] in ADAPTER_STATUSES, f"{path}.status", "invalid status")
    _id(adapter["adapter"], f"{path}.adapter")
    reason = adapter["reason"]
    _require(isinstance(reason, str), f"{path}.reason", "expected a string")
    _str(adapter["startup_compile_contract"], f"{path}.startup_compile_contract")
    if adapter["status"] == "ready":
        _require(
            not reason, f"{path}.reason", "ready adapters must have an empty reason"
        )
        _argv(adapter["command_argv"], f"{path}.command_argv")
        _argv(adapter["dispatch_argv"], f"{path}.dispatch_argv")
    else:
        _require(bool(reason), f"{path}.reason", "blocked adapters require a reason")
        _require(
            adapter["command_argv"] in (None, []),
            f"{path}.command_argv",
            "blocked adapters cannot advertise a command",
        )
        _require(
            adapter["dispatch_argv"] in (None, []),
            f"{path}.dispatch_argv",
            "blocked adapters cannot advertise a dispatch command",
        )
    if scope == "forward_backward" and adapter["status"] == "ready":
        _require(
            "direct" in adapter["startup_compile_contract"].lower(),
            path,
            "combined adapter must state direct measurement",
        )
    return adapter


def write_frozen_manifest(manifest: Mapping[str, Any], output: Path) -> str:
    validated = validate_manifest(dict(manifest))
    payload = (canonical_json(validated) + "\n").encode("ascii")
    digest = sha256_bytes(payload)
    sidecar = output.with_suffix(output.suffix + ".sha256")
    expected_sidecar = f"{digest}  {output.name}\n"
    if output.exists() or sidecar.exists():
        _require(
            output.is_file() and sidecar.is_file(),
            str(output),
            "partial frozen manifest exists",
        )
        _require(
            output.read_bytes() == payload,
            str(output),
            "refusing to overwrite different manifest",
        )
        _require(
            sidecar.read_text(encoding="ascii") == expected_sidecar,
            str(sidecar),
            "hash sidecar differs",
        )
        return digest
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    sidecar.write_text(expected_sidecar, encoding="ascii")
    return digest


def load_frozen_manifest(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
        manifest = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"cannot load manifest {path}: {exc}") from exc
    canonical_payload = (canonical_json(manifest) + "\n").encode("ascii")
    _require(
        payload == canonical_payload, str(path), "manifest is not canonical/frozen"
    )
    digest = sha256_bytes(payload)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    try:
        sidecar_value = sidecar.read_text(encoding="ascii")
    except OSError as exc:
        raise HarnessError(
            f"cannot read manifest hash sidecar {sidecar}: {exc}"
        ) from exc
    _require(
        sidecar_value == f"{digest}  {path.name}\n", str(sidecar), "manifest hash drift"
    )
    return validate_manifest(manifest), digest


def _artifact_by_name(manifest: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return next(item for item in manifest["frozen_artifacts"] if item["name"] == name)


def derive_run_plan(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
    events: Sequence[Mapping[str, Any]] = (),
    retry_statuses: Iterable[str] = (),
) -> list[dict[str, Any]]:
    validate_manifest(manifest)
    retry = set(retry_statuses)
    _require(retry <= set(TERMINAL_STATUSES), "retry_statuses", "invalid status")
    event_index = _event_index(events, manifest_sha256)
    protocol = manifest["protocol"]
    root = Path(manifest["artifact_root"])
    plan: list[dict[str, Any]] = []
    sequence = 0
    for dataset in manifest["datasets"]:
        for scope in protocol["scopes"]:
            adapter = manifest["adapters"][scope]
            for repeat_index, (seed, order) in enumerate(
                zip(protocol["seeds"], protocol["pair_order"])
            ):
                pair_id = f"{manifest['study_id']}.{dataset['scene_id']}.{scope}.r{repeat_index:02d}.s{seed}"
                side_order = (
                    SIDES if order == "baseline-first" else tuple(reversed(SIDES))
                )
                for side in side_order:
                    logical_sequence = sequence
                    sequence += 1
                    run_id = f"{pair_id}.{side}"
                    prior = event_index.get(run_id, [])
                    latest = prior[-1] if prior else None
                    if latest is not None and latest["status"] not in retry:
                        continue
                    attempt_index = (
                        latest["attempt_index"] + 1 if latest is not None else 0
                    )
                    attempt_id = f"{run_id}.a{attempt_index:03d}"
                    output_path = (
                        root
                        / "runs"
                        / run_id
                        / f"attempt-{attempt_index:03d}"
                        / "raw.json"
                    )
                    dispatch_path = output_path.with_name("dispatch.json")
                    checkpoint_path = (
                        root
                        / "checkpoints"
                        / dataset["scene_id"]
                        / f"seed-{seed}"
                        / "step-006999.pt"
                    )
                    values = {
                        "adapter": adapter["adapter"],
                        "artifact_root": root.as_posix(),
                        "attempt_id": attempt_id,
                        "attempt_index": str(attempt_index),
                        "backend": manifest["variants"][side]["backend"],
                        "checkpoint_path": checkpoint_path.as_posix(),
                        "data_factor": str(dataset["data_factor"]),
                        "dataset_path": dataset["path"],
                        "dispatch_path": dispatch_path.as_posix(),
                        "manifest_path": manifest_path.as_posix(),
                        "manifest_sha256": manifest_sha256,
                        "measured_iterations": str(protocol["measured_iterations"]),
                        "output_path": output_path.as_posix(),
                        "pair_id": pair_id,
                        "rounds": str(protocol["rounds"]),
                        "run_id": run_id,
                        "scene_id": dataset["scene_id"],
                        "scope": scope,
                        "seed": str(seed),
                        "side": side,
                        "training_steps": str(protocol["training_steps"]),
                        "variant": manifest["variants"][side]["label"],
                        "warmup_iterations": str(protocol["warmup_iterations"]),
                    }
                    command = (
                        _expand_argv(adapter["command_argv"], values)
                        if adapter["status"] == "ready"
                        else None
                    )
                    dispatch_command = (
                        _expand_argv(adapter["dispatch_argv"], values)
                        if adapter["status"] == "ready"
                        else None
                    )
                    plan.append(
                        {
                            "sequence_index": logical_sequence,
                            "run_id": run_id,
                            "attempt_id": attempt_id,
                            "attempt_index": attempt_index,
                            "pair_id": pair_id,
                            "repeat_index": repeat_index,
                            "pair_order": order,
                            "scene_id": dataset["scene_id"],
                            "scope": scope,
                            "seed": seed,
                            "side": side,
                            "backend": manifest["variants"][side]["backend"],
                            "adapter": adapter["adapter"],
                            "adapter_status": adapter["status"],
                            "blocked_reason": adapter["reason"] or None,
                            "command_argv": command,
                            "dispatch_argv": dispatch_command,
                            "output_path": output_path.as_posix(),
                            "dispatch_path": dispatch_path.as_posix(),
                            "dataset_manifest_sha256": dataset["manifest_sha256"],
                            "artifact_sha256": _artifact_by_name(
                                manifest, manifest["variants"][side]["artifact"]
                            )["sha256"],
                        }
                    )
    return plan


def _expand_argv(
    template: Sequence[str] | None, values: Mapping[str, str]
) -> list[str]:
    _require(template is not None, "template", "missing argv")
    result: list[str] = []
    for token in template:
        result.append(_PLACEHOLDER.sub(lambda match: values[match.group(1)], token))
    return result


def format_plan(
    plan: Sequence[Mapping[str, Any]], manifest_path: Path, events_path: Path
) -> str:
    lines: list[str] = []
    for item in plan:
        if item["adapter_status"] == "blocked":
            lines.append(f"BLOCKED {item['attempt_id']} :: {item['blocked_reason']}")
            continue
        wrapper = [
            "python3",
            "-m",
            "benchmarks.performance_study",
            "execute",
            "--manifest",
            manifest_path.as_posix(),
            "--events",
            events_path.as_posix(),
            "--run-id",
            item["run_id"],
        ]
        lines.append(f"RUN {item['attempt_id']} :: {shlex.join(wrapper)}")
        lines.append(f"  dispatch :: {shlex.join(item['dispatch_argv'])}")
        lines.append(f"  executor :: {shlex.join(item['command_argv'])}")
    return "\n".join(lines) + ("\n" if lines else "")


@dataclass(frozen=True)
class ProbeSnapshot:
    viewer_states: Mapping[str, str]
    gpu_compute_pids: tuple[int, ...]
    tracked_tree_clean: bool
    head_commit: str
    tracked_tree_sha256: str
    container_image_id: str
    dataset_manifest_sha256: Mapping[str, str]
    artifact_sha256: Mapping[str, str]

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "ProbeSnapshot":
        obj = _dict(dict(value), "probe")
        _exact_keys(
            obj,
            (
                "viewer_states",
                "gpu_compute_pids",
                "tracked_tree_clean",
                "head_commit",
                "tracked_tree_sha256",
                "container_image_id",
                "dataset_manifest_sha256",
                "artifact_sha256",
            ),
            (),
            "probe",
        )
        viewers = _dict(obj["viewer_states"], "probe.viewer_states")
        pids = tuple(
            _int(pid, "probe.gpu_compute_pids[]", 1)
            for pid in _list(obj["gpu_compute_pids"], "probe.gpu_compute_pids")
        )
        datasets = _dict(
            obj["dataset_manifest_sha256"], "probe.dataset_manifest_sha256"
        )
        artifacts = _dict(obj["artifact_sha256"], "probe.artifact_sha256")
        _require(
            isinstance(obj["tracked_tree_clean"], bool),
            "probe.tracked_tree_clean",
            "expected boolean",
        )
        return cls(
            viewer_states={
                str(key): _str(value, f"probe.viewer_states.{key}")
                for key, value in viewers.items()
            },
            gpu_compute_pids=pids,
            tracked_tree_clean=obj["tracked_tree_clean"],
            head_commit=_hex(obj["head_commit"], "probe.head_commit", 40),
            tracked_tree_sha256=_hex(
                obj["tracked_tree_sha256"], "probe.tracked_tree_sha256", 64
            ),
            container_image_id=_str(
                obj["container_image_id"], "probe.container_image_id"
            ),
            dataset_manifest_sha256={
                str(key): _hex(value, f"probe.dataset_manifest_sha256.{key}", 64)
                for key, value in datasets.items()
            },
            artifact_sha256={
                str(key): _hex(value, f"probe.artifact_sha256.{key}", 64)
                for key, value in artifacts.items()
            },
        )


def check_preflight(
    manifest: Mapping[str, Any], snapshot: ProbeSnapshot
) -> list[dict[str, Any]]:
    validate_manifest(manifest)
    expected_environment = manifest["environment"]
    gates: list[dict[str, Any]] = []

    def add(name: str, passed: bool, evidence: str) -> None:
        gates.append({"name": name, "passed": bool(passed), "evidence": evidence})

    running = [
        name
        for name in expected_environment["viewer_containers"]
        if snapshot.viewer_states.get(name, "absent") == "running"
    ]
    add("viewer_containers_stopped", not running, "running=" + ",".join(running))
    add(
        "gpu_compute_exclusive",
        not snapshot.gpu_compute_pids,
        f"pids={list(snapshot.gpu_compute_pids)}",
    )
    add(
        "tracked_tree_clean",
        snapshot.tracked_tree_clean,
        f"clean={snapshot.tracked_tree_clean}",
    )
    add(
        "evaluator_commit_pinned",
        snapshot.head_commit == manifest["repository"]["evaluator_commit"],
        f"actual={snapshot.head_commit}",
    )
    add(
        "tracked_tree_pinned",
        snapshot.tracked_tree_sha256 == manifest["repository"]["tracked_tree_sha256"],
        f"actual={snapshot.tracked_tree_sha256}",
    )
    add(
        "container_image_pinned",
        snapshot.container_image_id == expected_environment["container_image_id"],
        f"actual={snapshot.container_image_id}",
    )
    for dataset in manifest["datasets"]:
        actual = snapshot.dataset_manifest_sha256.get(dataset["scene_id"])
        add(
            f"dataset_manifest.{dataset['scene_id']}",
            actual == dataset["manifest_sha256"],
            f"actual={actual}",
        )
    for artifact in manifest["frozen_artifacts"]:
        actual = snapshot.artifact_sha256.get(artifact["name"])
        add(
            f"artifact.{artifact['name']}",
            actual == artifact["sha256"],
            f"actual={actual}",
        )
    add(
        "startup_compile_exclusion_contract",
        manifest["protocol"]["startup_compile_excluded"] is True
        and bool(manifest["protocol"]["startup_compile_contract"]),
        manifest["protocol"]["startup_compile_contract"],
    )
    add(
        "paired_order_frozen",
        len(manifest["protocol"]["pair_order"]) == len(manifest["protocol"]["seeds"]),
        canonical_json(manifest["protocol"]["pair_order"]),
    )
    return gates


def collect_live_probe(manifest: Mapping[str, Any], repo_root: Path) -> ProbeSnapshot:
    """Collect host-only gate state. This does not start containers or CUDA work."""

    environment = manifest["environment"]
    viewer_states: dict[str, str] = {}
    for name in environment["viewer_containers"]:
        completed = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Status}}", name],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        viewer_states[name] = (
            completed.stdout.strip() if completed.returncode == 0 else "absent"
        )
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
            "-i",
            str(environment["gpu_index"]),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    pids = tuple(int(line.strip()) for line in gpu.stdout.splitlines() if line.strip())
    status = _run_git(repo_root, "status", "--porcelain=v1", "--untracked-files=no")
    head = _resolve_commit(repo_root, "HEAD")
    image = subprocess.run(
        [
            "docker",
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            environment["container_image"],
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return ProbeSnapshot(
        viewer_states=viewer_states,
        gpu_compute_pids=pids,
        tracked_tree_clean=not bool(status),
        head_commit=head,
        tracked_tree_sha256=_tracked_tree_hash(repo_root, head),
        container_image_id=image,
        dataset_manifest_sha256={
            item["scene_id"]: sha256_file(repo_root / item["manifest_path"])
            for item in manifest["datasets"]
        },
        artifact_sha256={
            item["name"]: sha256_file(repo_root / item["path"])
            for item in manifest["frozen_artifacts"]
        },
    )


def validate_raw_run(
    value: Any,
    planned: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_sha256: str,
) -> dict[str, Any]:
    raw = _dict(value, "raw")
    required = (
        "schema_version",
        "record_type",
        "manifest_sha256",
        "run_id",
        "attempt_id",
        "attempt_index",
        "pair_id",
        "sequence_index",
        "scene_id",
        "scope",
        "seed",
        "side",
        "status",
        "started_at_utc",
        "completed_at_utc",
        "command_argv",
        "gates",
        "dispatch_proof",
        "timing",
        "telemetry",
        "correctness",
        "artifact_hashes",
        "rejection_reasons",
    )
    _exact_keys(raw, required, (), "raw")
    _require(raw["schema_version"] == 1, "raw.schema_version", "expected 1")
    _require(
        raw["record_type"] == RAW_RUN_RECORD_TYPE, "raw.record_type", "invalid type"
    )
    _require(
        raw["manifest_sha256"] == manifest_sha256,
        "raw.manifest_sha256",
        "manifest hash drift",
    )
    for key in (
        "run_id",
        "attempt_id",
        "attempt_index",
        "pair_id",
        "sequence_index",
        "scene_id",
        "scope",
        "seed",
        "side",
    ):
        _require(
            raw[key] == planned[key],
            f"raw.{key}",
            f"does not match plan ({planned[key]!r})",
        )
    _require(
        raw["status"] in TERMINAL_STATUSES, "raw.status", "invalid terminal status"
    )
    for key in ("started_at_utc", "completed_at_utc"):
        value_string = _str(raw[key], f"raw.{key}")
        _require(
            bool(_UTC.fullmatch(value_string)),
            f"raw.{key}",
            "expected UTC timestamp ending in Z",
        )
    command = _list(raw["command_argv"], "raw.command_argv")
    for index, token in enumerate(command):
        _str(token, f"raw.command_argv[{index}]")
    if planned["command_argv"] is not None:
        _require(
            command == planned["command_argv"],
            "raw.command_argv",
            "does not match frozen command",
        )

    gates = _list(raw["gates"], "raw.gates")
    gate_names: set[str] = set()
    for index, gate_raw in enumerate(gates):
        gate = _dict(gate_raw, f"raw.gates[{index}]")
        _exact_keys(gate, ("name", "passed", "evidence"), (), f"raw.gates[{index}]")
        name = _str(gate["name"], f"raw.gates[{index}].name")
        _require(name not in gate_names, f"raw.gates[{index}].name", "duplicate")
        gate_names.add(name)
        _require(
            isinstance(gate["passed"], bool),
            f"raw.gates[{index}].passed",
            "expected boolean",
        )
        _str(gate["evidence"], f"raw.gates[{index}].evidence")

    dispatch = _dict(raw["dispatch_proof"], "raw.dispatch_proof")
    _exact_keys(
        dispatch,
        ("requested_backend", "resolved_backend", "artifact_sha256", "proof_sha256"),
        (),
        "raw.dispatch_proof",
    )
    expected_variant = manifest["variants"][planned["side"]]
    _str(dispatch["requested_backend"], "raw.dispatch_proof.requested_backend")
    _str(dispatch["resolved_backend"], "raw.dispatch_proof.resolved_backend")
    _hex(dispatch["artifact_sha256"], "raw.dispatch_proof.artifact_sha256", 64)
    _hex(dispatch["proof_sha256"], "raw.dispatch_proof.proof_sha256", 64)

    timing = _dict(raw["timing"], "raw.timing")
    _exact_keys(
        timing,
        (
            "unit",
            "samples",
            "warmup_iterations",
            "measured_iterations",
            "rounds",
            "training_steps",
            "startup_compile_excluded",
            "direct_measurement",
        ),
        (),
        "raw.timing",
    )
    expected_unit = "s" if planned["scope"] == "training" else "ms"
    _require(
        timing["unit"] == expected_unit, "raw.timing.unit", f"expected {expected_unit}"
    )
    samples = _list(timing["samples"], "raw.timing.samples")
    for index, sample in enumerate(samples):
        _float(sample, f"raw.timing.samples[{index}]", positive=True)
    protocol = manifest["protocol"]
    for key in ("warmup_iterations", "measured_iterations", "rounds", "training_steps"):
        _require(
            timing[key] == protocol[key], f"raw.timing.{key}", "does not match protocol"
        )
    _require(
        timing["startup_compile_excluded"] is True,
        "raw.timing.startup_compile_excluded",
        "must be true",
    )
    _require(
        isinstance(timing["direct_measurement"], bool),
        "raw.timing.direct_measurement",
        "expected boolean",
    )
    if planned["scope"] == "forward_backward":
        _require(
            timing["direct_measurement"] is True,
            "raw.timing.direct_measurement",
            "forward+backward must be directly measured",
        )
    if raw["status"] == "complete":
        expected_sample_count = (
            1 if planned["scope"] == "training" else protocol["rounds"]
        )
        _require(
            len(samples) == expected_sample_count,
            "raw.timing.samples",
            f"expected {expected_sample_count} samples",
        )

    telemetry = _dict(raw["telemetry"], "raw.telemetry")
    _exact_keys(
        telemetry,
        (
            "temperature_start_c",
            "temperature_end_c",
            "sm_clock_samples_mhz",
            "memory_clock_samples_mhz",
            "power_samples_w",
            "active_gpu_compute_pids_before",
            "active_gpu_compute_pids_after",
            "throttle_reasons",
            "xid_errors",
        ),
        (),
        "raw.telemetry",
    )
    for key in ("temperature_start_c", "temperature_end_c"):
        _float(telemetry[key], f"raw.telemetry.{key}")
    for key in ("sm_clock_samples_mhz", "memory_clock_samples_mhz", "power_samples_w"):
        values = _list(telemetry[key], f"raw.telemetry.{key}")
        if raw["status"] == "complete":
            _require(
                bool(values), f"raw.telemetry.{key}", "complete runs require telemetry"
            )
        for index, sample in enumerate(values):
            _float(sample, f"raw.telemetry.{key}[{index}]", positive=True)
    for key in ("active_gpu_compute_pids_before", "active_gpu_compute_pids_after"):
        values = _list(telemetry[key], f"raw.telemetry.{key}")
        for index, pid in enumerate(values):
            _int(pid, f"raw.telemetry.{key}[{index}]", 1)
    for key in ("throttle_reasons", "xid_errors"):
        values = _list(telemetry[key], f"raw.telemetry.{key}")
        for index, item in enumerate(values):
            _str(item, f"raw.telemetry.{key}[{index}]")

    correctness = _dict(raw["correctness"], "raw.correctness")
    _exact_keys(
        correctness, ("passed", "gate", "artifact_sha256"), (), "raw.correctness"
    )
    _require(
        isinstance(correctness["passed"], bool),
        "raw.correctness.passed",
        "expected boolean",
    )
    _str(correctness["gate"], "raw.correctness.gate")
    if correctness["artifact_sha256"] is not None:
        _hex(correctness["artifact_sha256"], "raw.correctness.artifact_sha256", 64)

    hashes = _dict(raw["artifact_hashes"], "raw.artifact_hashes")
    for name, digest in hashes.items():
        _id(name, "raw.artifact_hashes key")
        _hex(digest, f"raw.artifact_hashes.{name}", 64)
    reasons = _list(raw["rejection_reasons"], "raw.rejection_reasons")
    for index, reason in enumerate(reasons):
        _str(reason, f"raw.rejection_reasons[{index}]")

    if raw["status"] == "complete":
        _require(
            all(gate["passed"] for gate in gates),
            "raw.gates",
            "complete run has failed gate",
        )
        _require(
            dispatch["requested_backend"] == expected_variant["backend"],
            "raw.dispatch_proof.requested_backend",
            "backend mismatch",
        )
        _require(
            dispatch["resolved_backend"] == expected_variant["expected_dispatch"],
            "raw.dispatch_proof.resolved_backend",
            "dispatch fallback/mismatch",
        )
        _require(
            dispatch["artifact_sha256"] == planned["artifact_sha256"],
            "raw.dispatch_proof.artifact_sha256",
            "artifact drift",
        )
        _require(
            correctness["passed"] and correctness["artifact_sha256"] is not None,
            "raw.correctness",
            "complete run requires passing correctness",
        )
        _require(
            not reasons,
            "raw.rejection_reasons",
            "complete run cannot have rejection reasons",
        )
    else:
        _require(bool(reasons), "raw.rejection_reasons", "failed runs require a reason")
    return raw


def _event_index(
    events: Sequence[Mapping[str, Any]], manifest_sha256: str
) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    attempt_ids: set[str] = set()
    for position, raw in enumerate(events):
        event = _dict(raw, f"events[{position}]")
        _exact_keys(
            event,
            (
                "schema_version",
                "record_type",
                "event_id",
                "manifest_sha256",
                "run_id",
                "attempt_id",
                "attempt_index",
                "status",
                "raw_sha256",
                "raw_artifact",
            ),
            (),
            f"events[{position}]",
        )
        _require(
            event["schema_version"] == 1,
            f"events[{position}].schema_version",
            "expected 1",
        )
        _require(
            event["record_type"] == RUN_EVENT_RECORD_TYPE,
            f"events[{position}].record_type",
            "invalid type",
        )
        _require(
            event["manifest_sha256"] == manifest_sha256,
            f"events[{position}].manifest_sha256",
            "manifest hash drift",
        )
        attempt_id = _str(event["attempt_id"], f"events[{position}].attempt_id")
        _require(
            event["event_id"] == attempt_id,
            f"events[{position}].event_id",
            "must equal attempt_id",
        )
        _require(
            attempt_id not in attempt_ids, f"events[{position}].attempt_id", "duplicate"
        )
        attempt_ids.add(attempt_id)
        _int(event["attempt_index"], f"events[{position}].attempt_index")
        _require(
            event["status"] in TERMINAL_STATUSES,
            f"events[{position}].status",
            "invalid status",
        )
        _hex(event["raw_sha256"], f"events[{position}].raw_sha256", 64)
        _relative(event["raw_artifact"], f"events[{position}].raw_artifact")
        index.setdefault(
            _str(event["run_id"], f"events[{position}].run_id"), []
        ).append(dict(event))
    for run_id, attempts in index.items():
        attempts.sort(key=lambda item: item["attempt_index"])
        expected = list(range(len(attempts)))
        actual = [item["attempt_index"] for item in attempts]
        _require(
            actual == expected,
            run_id,
            f"attempt indexes must be contiguous, got {actual}",
        )
    return index


def load_run_events(path: Path, manifest_sha256: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise HarnessError(
                f"{path}:{line_number}: invalid JSON: {exc.msg}"
            ) from exc
    _event_index(events, manifest_sha256)
    return events


def append_run_event(
    path: Path, raw: Mapping[str, Any], raw_artifact: Path
) -> tuple[dict[str, Any], bool]:
    raw_payload = (canonical_json(raw) + "\n").encode("ascii")
    event = {
        "schema_version": 1,
        "record_type": RUN_EVENT_RECORD_TYPE,
        "event_id": raw["attempt_id"],
        "manifest_sha256": raw["manifest_sha256"],
        "run_id": raw["run_id"],
        "attempt_id": raw["attempt_id"],
        "attempt_index": raw["attempt_index"],
        "status": raw["status"],
        "raw_sha256": sha256_bytes(raw_payload),
        "raw_artifact": raw_artifact.as_posix(),
    }
    existing = load_run_events(path, raw["manifest_sha256"])
    matches = [item for item in existing if item["attempt_id"] == event["attempt_id"]]
    if matches:
        _require(
            matches[0] == event,
            event["attempt_id"],
            "duplicate ID has different content",
        )
        return event, False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="ascii") as handle:
        handle.write(canonical_json(event) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return event, True


def import_raw_artifacts(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
    events_path: Path,
    raw_paths: Sequence[Path],
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    events = load_run_events(events_path, manifest_sha256)
    imported = 0
    idempotent = 0
    for raw_path in raw_paths:
        try:
            raw_value = json.loads(raw_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HarnessError(f"cannot read raw artifact {raw_path}: {exc}") from exc
        attempt_id = (
            raw_value.get("attempt_id") if isinstance(raw_value, dict) else None
        )
        run_id = raw_value.get("run_id") if isinstance(raw_value, dict) else None
        attempt_index = (
            raw_value.get("attempt_index") if isinstance(raw_value, dict) else None
        )
        _require(
            isinstance(run_id, str) and isinstance(attempt_index, int),
            str(raw_path),
            "missing run identity",
        )
        planned = _plan_for_attempt(
            manifest, manifest_path, manifest_sha256, run_id, attempt_index
        )
        _require(
            attempt_id == planned["attempt_id"],
            str(raw_path),
            "attempt is absent from frozen plan",
        )
        prior = _event_index(events, manifest_sha256).get(run_id, [])
        _require(
            len(prior) == attempt_index,
            str(raw_path),
            "prior attempts are missing or retry index is stale",
        )
        expected_path = Path(planned["output_path"])
        if not expected_path.is_absolute():
            expected_path = repo_root / expected_path
        _require(
            raw_path.resolve() == expected_path.resolve(),
            str(raw_path),
            "raw artifact is outside its idempotent planned path",
        )
        raw = validate_raw_run(raw_value, planned, manifest, manifest_sha256)
        event_path = raw_path.resolve()
        try:
            event_path = event_path.relative_to(repo_root.resolve())
        except ValueError as exc:
            raise HarnessError(
                f"{raw_path}: raw artifact must be inside repository root"
            ) from exc
        _, added = append_run_event(events_path, raw, event_path)
        imported += int(added)
        idempotent += int(not added)
        events = load_run_events(events_path, manifest_sha256)
    updated = load_run_events(events_path, manifest_sha256)
    complete, summary = terminal_coverage(
        manifest, manifest_path, manifest_sha256, updated
    )
    return {
        "imported": imported,
        "idempotent": idempotent,
        "terminal_coverage": complete,
        "coverage": summary,
    }


def _plan_for_attempt(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
    run_id: str,
    attempt_index: int,
) -> dict[str, Any]:
    _int(attempt_index, "attempt_index")
    synthetic_prior = [
        {
            "schema_version": 1,
            "record_type": RUN_EVENT_RECORD_TYPE,
            "event_id": f"{run_id}.a{index:03d}",
            "manifest_sha256": manifest_sha256,
            "run_id": run_id,
            "attempt_id": f"{run_id}.a{index:03d}",
            "attempt_index": index,
            "status": "rejected",
            "raw_sha256": "0" * 64,
            "raw_artifact": f"synthetic/{index}.json",
        }
        for index in range(attempt_index)
    ]
    plan = derive_run_plan(
        manifest,
        manifest_path,
        manifest_sha256,
        synthetic_prior,
        retry_statuses=TERMINAL_STATUSES,
    )
    matches = [item for item in plan if item["run_id"] == run_id]
    _require(len(matches) == 1, run_id, "run is absent from frozen plan")
    return matches[0]


def terminal_coverage(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
    events: Sequence[Mapping[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    expected = derive_run_plan(manifest, manifest_path, manifest_sha256, ())
    index = _event_index(events, manifest_sha256)
    missing = [item["run_id"] for item in expected if item["run_id"] not in index]
    latest = [attempts[-1] for attempts in index.values()]
    counts = {
        status: sum(item["status"] == status for item in latest)
        for status in TERMINAL_STATUSES
    }
    return not missing, {
        "expected": len(expected),
        "terminal": len(index),
        "missing": missing,
        "latest_status_counts": counts,
    }


def _load_raw_for_event(event: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    path = Path(event["raw_artifact"])
    if not path.is_absolute():
        path = repo_root / path
    try:
        payload = path.read_bytes()
        raw = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"cannot load event artifact {path}: {exc}") from exc
    _require(
        sha256_bytes((canonical_json(raw) + "\n").encode("ascii"))
        == event["raw_sha256"],
        str(path),
        "raw artifact hash drift",
    )
    return raw


def _pair_rejection_reasons(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> list[str]:
    reasons: list[str] = []
    b_tel = baseline["telemetry"]
    c_tel = candidate["telemetry"]
    if b_tel["temperature_start_c"] > 85 or c_tel["temperature_start_c"] > 85:
        reasons.append("temperature_start_above_85c")
    if abs(b_tel["temperature_start_c"] - c_tel["temperature_start_c"]) > 3:
        reasons.append("paired_start_temperature_delta_above_3c")
    for field, label in (
        ("sm_clock_samples_mhz", "sm_clock"),
        ("memory_clock_samples_mhz", "memory_clock"),
    ):
        b_clock = statistics.median(b_tel[field])
        c_clock = statistics.median(c_tel[field])
        if abs(b_clock - c_clock) / b_clock > 0.01:
            reasons.append(f"paired_{label}_delta_above_1pct")
    for side, telemetry in (("baseline", b_tel), ("candidate", c_tel)):
        if (
            telemetry["active_gpu_compute_pids_before"]
            or telemetry["active_gpu_compute_pids_after"]
        ):
            reasons.append(f"{side}_gpu_process_contamination")
        if telemetry["throttle_reasons"]:
            reasons.append(f"{side}_throttling")
        if telemetry["xid_errors"]:
            reasons.append(f"{side}_xid_error")
    return sorted(set(reasons))


def build_result_record(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
    events: Sequence[Mapping[str, Any]],
    repo_root: Path,
    attempt_index: int,
) -> dict[str, Any]:
    complete_coverage, coverage = terminal_coverage(
        manifest, manifest_path, manifest_sha256, events
    )
    _require(
        complete_coverage,
        "events",
        f"incomplete terminal coverage ({len(coverage['missing'])} missing)",
    )
    index = _event_index(events, manifest_sha256)
    planned = derive_run_plan(manifest, manifest_path, manifest_sha256, ())
    latest_events = {run_id: attempts[-1] for run_id, attempts in index.items()}
    raw_by_run: dict[str, dict[str, Any]] = {}
    for item in planned:
        event = latest_events[item["run_id"]]
        raw = _load_raw_for_event(event, repo_root)
        actual_plan = _plan_for_attempt(
            manifest,
            manifest_path,
            manifest_sha256,
            item["run_id"],
            event["attempt_index"],
        )
        raw_by_run[item["run_id"]] = validate_raw_run(
            raw, actual_plan, manifest, manifest_sha256
        )

    statuses = {raw["status"] for raw in raw_by_run.values()}
    status = (
        "crash"
        if "crash" in statuses
        else "discard"
        if "rejected" in statuses
        else "keep"
    )
    pair_rejections: list[str] = []
    measurements: list[dict[str, Any]] = []
    for dataset in manifest["datasets"]:
        for scope in manifest["protocol"]["scopes"]:
            baseline_samples: list[float] = []
            candidate_samples: list[float] = []
            group_complete = True
            for repeat_index, seed in enumerate(manifest["protocol"]["seeds"]):
                pair_id = f"{manifest['study_id']}.{dataset['scene_id']}.{scope}.r{repeat_index:02d}.s{seed}"
                baseline = raw_by_run[f"{pair_id}.baseline"]
                candidate = raw_by_run[f"{pair_id}.candidate"]
                if (
                    baseline["status"] != "complete"
                    or candidate["status"] != "complete"
                ):
                    group_complete = False
                    continue
                reasons = _pair_rejection_reasons(baseline, candidate)
                if reasons:
                    pair_rejections.extend(f"{pair_id}:{reason}" for reason in reasons)
                baseline_samples.append(
                    statistics.median(baseline["timing"]["samples"])
                )
                candidate_samples.append(
                    statistics.median(candidate["timing"]["samples"])
                )
            if group_complete and len(baseline_samples) == len(
                manifest["protocol"]["seeds"]
            ):
                measurements.append(
                    {
                        "scene_id": dataset["scene_id"],
                        "scope": scope,
                        "unit": "s" if scope == "training" else "ms",
                        "baseline_samples": baseline_samples,
                        "candidate_samples": candidate_samples,
                    }
                )
    if pair_rejections and status == "keep":
        status = "discard"
    if status != "crash":
        progress_key = (
            manifest["protocol"]["progress_scene_id"],
            manifest["protocol"]["progress_scope"],
        )
        if progress_key not in {
            (item["scene_id"], item["scope"]) for item in measurements
        }:
            status = "crash"

    complete_raw = [raw for raw in raw_by_run.values() if raw["status"] == "complete"]
    sm_clocks = [
        sample
        for raw in complete_raw
        for sample in raw["telemetry"]["sm_clock_samples_mhz"]
    ]
    memory_clocks = [
        sample
        for raw in complete_raw
        for sample in raw["telemetry"]["memory_clock_samples_mhz"]
    ]
    artifact_hash = (
        canonical_hash(
            sorted(
                (raw["attempt_id"], raw["correctness"]["artifact_sha256"])
                for raw in complete_raw
                if raw["correctness"]["artifact_sha256"] is not None
            )
        )
        if complete_raw
        else None
    )
    timestamps = sorted(raw["completed_at_utc"] for raw in raw_by_run.values())
    correctness_passed = status == "keep" and all(
        raw["correctness"]["passed"] for raw in complete_raw
    )
    record = {
        "schema_version": 1,
        "record_type": "ptxsplat.performance_history.result",
        "study_id": manifest["study_id"],
        "experiment_id": manifest["experiment_id"],
        "attempt_index": attempt_index,
        "timestamp_utc": timestamps[-1],
        "status": status,
        "description": manifest["description"]
        + (f"; rejected pairs: {len(pair_rejections)}" if pair_rejections else ""),
        "candidate": {
            "commit": manifest["variants"]["candidate"]["commit"],
            "label": manifest["variants"]["candidate"]["label"],
        },
        "baseline": {
            "commit": manifest["variants"]["baseline"]["commit"],
            "label": manifest["variants"]["baseline"]["label"],
        },
        "comparison_kind": manifest["comparison_kind"],
        "publication": bool(manifest["protocol"]["publication"] and status == "keep"),
        "promoted": False,
        "environment": {
            "environment_id": manifest["environment"]["environment_id"],
            "gpu_name": manifest["environment"]["gpu_name"],
            "gpu_uuid": manifest["environment"]["gpu_uuid"],
            "container_image": f"{manifest['environment']['container_image']}@{manifest['environment']['container_image_id']}",
            "driver_version": manifest["environment"]["driver_version"],
            "cuda_version": manifest["environment"]["cuda_version"],
            "pytorch_version": manifest["environment"]["pytorch_version"],
            "build_flags": manifest["environment"]["build_flags"],
            "sm_clock_mhz": statistics.median(sm_clocks)
            if sm_clocks
            else manifest["environment"]["sm_clock_mhz"],
            "memory_clock_mhz": statistics.median(memory_clocks)
            if memory_clocks
            else manifest["environment"]["memory_clock_mhz"],
        },
        "protocol": {
            "evaluator_commit": manifest["repository"]["evaluator_commit"],
            "evaluator_sha256": manifest["repository"]["evaluator_sha256"],
            "startup_compile_excluded": True,
            "seeds": manifest["protocol"]["seeds"],
            "pair_order": manifest["protocol"]["pair_order"],
            "warmup_iterations": manifest["protocol"]["warmup_iterations"],
            "measured_iterations": manifest["protocol"]["measured_iterations"],
            "rounds": manifest["protocol"]["rounds"],
            "training_steps": manifest["protocol"]["training_steps"],
            "progress_scene_id": manifest["protocol"]["progress_scene_id"],
            "progress_scope": manifest["protocol"]["progress_scope"],
        },
        "datasets": [
            {
                "scene_id": item["scene_id"],
                "dataset": item["dataset"],
                "sha256": item["manifest_sha256"],
                "data_factor": item["data_factor"],
            }
            for item in manifest["datasets"]
        ],
        "correctness": {
            "passed": correctness_passed,
            "gate": "all raw correctness, dispatch, contamination, thermal, and clock gates",
            "artifact_sha256": artifact_hash if correctness_passed else None,
        },
        "measurements": measurements if status != "crash" else [],
    }
    return validate_record(record)


def append_result_record(path: Path, record: Mapping[str, Any]) -> bool:
    validated = validate_record(dict(record))
    existing: list[dict[str, Any]] = []
    if path.exists():
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                existing.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise HarnessError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
    key = (
        validated["study_id"],
        validated["experiment_id"],
        validated["attempt_index"],
    )
    for item in existing:
        item_key = (
            item.get("study_id"),
            item.get("experiment_id"),
            item.get("attempt_index"),
        )
        if item_key == key:
            _require(
                item == validated, str(key), "duplicate result ID has different content"
            )
            return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="ascii") as handle:
        handle.write(canonical_json(validated) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return True


def utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
