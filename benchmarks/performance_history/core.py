from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence

REPORT_GENERATOR_VERSION = "1.0"
SCHEMA_VERSION = 1
RECORD_TYPE = "ptxsplat.performance_history.result"
SCOPES = (
    "isolated_forward",
    "isolated_backward",
    "forward_backward",
    "training",
)
STATUSES = ("keep", "discard", "crash")
PAIR_ORDERS = ("baseline-first", "candidate-first")
COMPARISON_KINDS = ("baseline", "historical", "ablation", "candidate")
ABLATION_METHODS = ("historical_checkpoint", "tip_removal")

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")


class ValidationError(ValueError):
    """Raised when a result cannot support a reproducible report."""


def _fail(path: str, message: str) -> None:
    raise ValidationError(f"{path}: {message}")


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "expected an object")
    return value


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(path, "expected an array")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(path, "expected a non-empty string")
    return value


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(path, f"expected an integer >= {minimum}")
    return value


def _number(value: Any, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "expected a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive " if positive else ""
        _fail(path, f"expected a finite {qualifier}number")
    return result


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(path, "expected a boolean")
    return value


def _required(obj: dict[str, Any], keys: Iterable[str], path: str) -> None:
    missing = sorted(set(keys) - set(obj))
    if missing:
        _fail(path, f"missing required fields: {', '.join(missing)}")


def _only(obj: dict[str, Any], keys: Iterable[str], path: str) -> None:
    extras = sorted(set(obj) - set(keys))
    if extras:
        _fail(path, f"unknown fields: {', '.join(extras)}")


def _validate_hex(value: Any, path: str, pattern: re.Pattern[str]) -> str:
    result = _string(value, path)
    if not pattern.fullmatch(result):
        expected_length = 40 if pattern is _HEX40 else 64
        _fail(path, f"expected {expected_length} lowercase hexadecimal characters")
    return result


def _validate_id(value: Any, path: str) -> str:
    result = _string(value, path)
    if not _ID.fullmatch(result):
        _fail(path, "must match [a-z0-9][a-z0-9._-]*")
    return result


def _validate_side(value: Any, path: str) -> dict[str, Any]:
    obj = _object(value, path)
    _required(obj, ("commit", "label"), path)
    _only(obj, ("commit", "label"), path)
    _validate_hex(obj["commit"], f"{path}.commit", _HEX40)
    _string(obj["label"], f"{path}.label")
    return obj


def _validate_environment(value: Any, path: str) -> dict[str, Any]:
    obj = _object(value, path)
    keys = (
        "environment_id",
        "gpu_name",
        "gpu_uuid",
        "container_image",
        "driver_version",
        "cuda_version",
        "pytorch_version",
        "build_flags",
        "sm_clock_mhz",
        "memory_clock_mhz",
    )
    _required(obj, keys, path)
    _only(obj, keys, path)
    _validate_id(obj["environment_id"], f"{path}.environment_id")
    for key in (
        "gpu_name",
        "gpu_uuid",
        "container_image",
        "driver_version",
        "cuda_version",
        "pytorch_version",
    ):
        _string(obj[key], f"{path}.{key}")
    flags = _list(obj["build_flags"], f"{path}.build_flags")
    if not flags:
        _fail(f"{path}.build_flags", "must not be empty")
    for index, flag in enumerate(flags):
        _string(flag, f"{path}.build_flags[{index}]")
    _number(obj["sm_clock_mhz"], f"{path}.sm_clock_mhz", positive=True)
    _number(obj["memory_clock_mhz"], f"{path}.memory_clock_mhz", positive=True)
    return obj


def _validate_protocol(value: Any, path: str, *, publication: bool) -> dict[str, Any]:
    obj = _object(value, path)
    keys = (
        "evaluator_commit",
        "evaluator_sha256",
        "startup_compile_excluded",
        "seeds",
        "pair_order",
        "warmup_iterations",
        "measured_iterations",
        "rounds",
        "training_steps",
        "progress_scene_id",
        "progress_scope",
    )
    _required(obj, keys, path)
    _only(obj, keys, path)
    _validate_hex(obj["evaluator_commit"], f"{path}.evaluator_commit", _HEX40)
    _validate_hex(obj["evaluator_sha256"], f"{path}.evaluator_sha256", _HEX64)
    if not _boolean(
        obj["startup_compile_excluded"], f"{path}.startup_compile_excluded"
    ):
        _fail(f"{path}.startup_compile_excluded", "must be true")
    seeds = _list(obj["seeds"], f"{path}.seeds")
    minimum_repeats = 5 if publication else 3
    if len(seeds) < minimum_repeats:
        _fail(
            f"{path}.seeds",
            f"requires at least {minimum_repeats} paired repeats",
        )
    normalized_seeds = [
        _integer(seed, f"{path}.seeds[{index}]") for index, seed in enumerate(seeds)
    ]
    if len(set(normalized_seeds)) != len(normalized_seeds):
        _fail(f"{path}.seeds", "repeat seeds must be unique")
    order = _list(obj["pair_order"], f"{path}.pair_order")
    if len(order) != len(seeds):
        _fail(f"{path}.pair_order", "must align one-to-one with seeds")
    for index, item in enumerate(order):
        if item not in PAIR_ORDERS:
            _fail(
                f"{path}.pair_order[{index}]",
                f"expected one of {', '.join(PAIR_ORDERS)}",
            )
    if abs(order.count(PAIR_ORDERS[0]) - order.count(PAIR_ORDERS[1])) > 1:
        _fail(f"{path}.pair_order", "baseline/candidate order is not balanced")
    _integer(obj["warmup_iterations"], f"{path}.warmup_iterations")
    _integer(obj["measured_iterations"], f"{path}.measured_iterations", minimum=1)
    _integer(obj["rounds"], f"{path}.rounds", minimum=1)
    _integer(obj["training_steps"], f"{path}.training_steps", minimum=1)
    _validate_id(obj["progress_scene_id"], f"{path}.progress_scene_id")
    if obj["progress_scope"] not in SCOPES:
        _fail(f"{path}.progress_scope", f"expected one of {', '.join(SCOPES)}")
    return obj


def _validate_datasets(value: Any, path: str) -> list[dict[str, Any]]:
    rows = _list(value, path)
    if not rows:
        _fail(path, "must list at least one scene")
    seen: set[str] = set()
    for index, item in enumerate(rows):
        item_path = f"{path}[{index}]"
        obj = _object(item, item_path)
        keys = ("scene_id", "dataset", "sha256", "data_factor")
        _required(obj, keys, item_path)
        _only(obj, keys, item_path)
        scene_id = _validate_id(obj["scene_id"], f"{item_path}.scene_id")
        if scene_id in seen:
            _fail(f"{item_path}.scene_id", "duplicate scene")
        seen.add(scene_id)
        _string(obj["dataset"], f"{item_path}.dataset")
        _validate_hex(obj["sha256"], f"{item_path}.sha256", _HEX64)
        _integer(obj["data_factor"], f"{item_path}.data_factor", minimum=1)
    return rows


def _validate_correctness(
    value: Any, path: str, *, status: str, publication: bool, promoted: bool
) -> dict[str, Any]:
    obj = _object(value, path)
    keys = ("passed", "gate", "artifact_sha256")
    _required(obj, keys, path)
    _only(obj, keys, path)
    passed = _boolean(obj["passed"], f"{path}.passed")
    _string(obj["gate"], f"{path}.gate")
    artifact = obj["artifact_sha256"]
    if artifact is not None:
        _validate_hex(artifact, f"{path}.artifact_sha256", _HEX64)
    if (publication or promoted or status == "keep") and not passed:
        _fail(f"{path}.passed", "keep/promoted/publication records must pass")
    if passed and artifact is None:
        _fail(f"{path}.artifact_sha256", "passed gates require an artifact hash")
    return obj


def _validate_measurements(
    value: Any,
    path: str,
    *,
    repeat_count: int,
    scenes: set[str],
) -> list[dict[str, Any]]:
    rows = _list(value, path)
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(rows):
        item_path = f"{path}[{index}]"
        obj = _object(item, item_path)
        keys = (
            "scene_id",
            "scope",
            "unit",
            "baseline_samples",
            "candidate_samples",
        )
        _required(obj, keys, item_path)
        _only(obj, keys, item_path)
        scene = _validate_id(obj["scene_id"], f"{item_path}.scene_id")
        if scene not in scenes:
            _fail(f"{item_path}.scene_id", "scene is absent from datasets")
        scope = obj["scope"]
        if scope not in SCOPES:
            _fail(f"{item_path}.scope", f"expected one of {', '.join(SCOPES)}")
        key = (scene, scope)
        if key in seen:
            _fail(item_path, f"duplicate measurement for {scene}/{scope}")
        seen.add(key)
        expected_unit = "s" if scope == "training" else "ms"
        if obj["unit"] != expected_unit:
            _fail(f"{item_path}.unit", f"{scope} requires unit {expected_unit!r}")
        for sample_key in ("baseline_samples", "candidate_samples"):
            samples = _list(obj[sample_key], f"{item_path}.{sample_key}")
            if len(samples) != repeat_count:
                _fail(
                    f"{item_path}.{sample_key}",
                    f"expected {repeat_count} values, one per paired repeat",
                )
            for sample_index, sample in enumerate(samples):
                _number(
                    sample,
                    f"{item_path}.{sample_key}[{sample_index}]",
                    positive=True,
                )
    return rows


def _validate_ablation(value: Any, path: str) -> dict[str, Any]:
    obj = _object(value, path)
    keys = ("group_id", "order", "component", "method")
    _required(obj, keys, path)
    _only(obj, keys, path)
    _validate_id(obj["group_id"], f"{path}.group_id")
    _integer(obj["order"], f"{path}.order")
    _string(obj["component"], f"{path}.component")
    if obj["method"] not in ABLATION_METHODS:
        _fail(f"{path}.method", f"expected one of {', '.join(ABLATION_METHODS)}")
    return obj


def validate_record(value: Any, path: str = "record") -> dict[str, Any]:
    record = _object(value, path)
    keys = (
        "schema_version",
        "record_type",
        "study_id",
        "experiment_id",
        "attempt_index",
        "timestamp_utc",
        "status",
        "description",
        "candidate",
        "baseline",
        "comparison_kind",
        "publication",
        "promoted",
        "environment",
        "protocol",
        "datasets",
        "correctness",
        "measurements",
        "ablation",
    )
    _required(record, set(keys) - {"ablation"}, path)
    _only(record, keys, path)
    if record["schema_version"] != SCHEMA_VERSION:
        _fail(f"{path}.schema_version", f"expected {SCHEMA_VERSION}")
    if record["record_type"] != RECORD_TYPE:
        _fail(f"{path}.record_type", f"expected {RECORD_TYPE!r}")
    _validate_id(record["study_id"], f"{path}.study_id")
    _validate_id(record["experiment_id"], f"{path}.experiment_id")
    _integer(record["attempt_index"], f"{path}.attempt_index")
    timestamp = _string(record["timestamp_utc"], f"{path}.timestamp_utc")
    if not _UTC.fullmatch(timestamp):
        _fail(f"{path}.timestamp_utc", "expected an ISO-8601 UTC timestamp ending in Z")
    status = record["status"]
    if status not in STATUSES:
        _fail(f"{path}.status", f"expected one of {', '.join(STATUSES)}")
    _string(record["description"], f"{path}.description")
    candidate = _validate_side(record["candidate"], f"{path}.candidate")
    baseline = _validate_side(record["baseline"], f"{path}.baseline")
    comparison_kind = record["comparison_kind"]
    if comparison_kind not in COMPARISON_KINDS:
        _fail(
            f"{path}.comparison_kind",
            f"expected one of {', '.join(COMPARISON_KINDS)}",
        )
    if comparison_kind == "baseline" and candidate != baseline:
        _fail(path, "baseline records require identical candidate and baseline")
    publication = _boolean(record["publication"], f"{path}.publication")
    promoted = _boolean(record["promoted"], f"{path}.promoted")
    if publication and status != "keep":
        _fail(f"{path}.publication", "only keep records may be published")
    if promoted and status != "keep":
        _fail(f"{path}.promoted", "only keep records may be promoted")
    if status == "crash" and (publication or promoted):
        _fail(path, "crash records cannot be promoted or published")
    _validate_environment(record["environment"], f"{path}.environment")
    protocol = _validate_protocol(
        record["protocol"], f"{path}.protocol", publication=publication
    )
    datasets = _validate_datasets(record["datasets"], f"{path}.datasets")
    scenes = {item["scene_id"] for item in datasets}
    if protocol["progress_scene_id"] not in scenes:
        _fail(
            f"{path}.protocol.progress_scene_id",
            "progress scene is absent from datasets",
        )
    _validate_correctness(
        record["correctness"],
        f"{path}.correctness",
        status=status,
        publication=publication,
        promoted=promoted,
    )
    measurements = _validate_measurements(
        record["measurements"],
        f"{path}.measurements",
        repeat_count=len(protocol["seeds"]),
        scenes=scenes,
    )
    if status != "crash":
        progress_key = (protocol["progress_scene_id"], protocol["progress_scope"])
        keys_present = {(item["scene_id"], item["scope"]) for item in measurements}
        if progress_key not in keys_present:
            _fail(
                f"{path}.measurements",
                f"missing immutable progress evaluator {progress_key[0]}/{progress_key[1]}",
            )
    if "ablation" in record:
        _validate_ablation(record["ablation"], f"{path}.ablation")
        if not publication:
            _fail(
                f"{path}.ablation", "ablation checkpoints must be publication records"
            )
    return record


def _records_from_json(value: Any, path: Path) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and set(value) == {"records"}:
        return _list(value["records"], f"{path}.records")
    if isinstance(value, dict):
        return [value]
    _fail(str(path), "JSON input must be a record, an array, or {'records': [...]}")


def load_records(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    records: list[Any] = []
    for raw_path in paths:
        path = Path(raw_path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValidationError(f"{path}: cannot read input: {exc}") from exc
        if path.suffix == ".jsonl":
            for line_number, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValidationError(
                        f"{path}:{line_number}: invalid JSON: {exc.msg}"
                    ) from exc
        else:
            try:
                records.extend(_records_from_json(json.loads(text), path))
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path}: invalid JSON: {exc.msg}") from exc
    if not records:
        raise ValidationError("no result records found")
    return [
        validate_record(record, f"records[{index}]")
        for index, record in enumerate(records)
    ]


def _dataset_signature(record: dict[str, Any]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        sorted(
            (
                item["scene_id"],
                item["dataset"],
                item["sha256"],
                item["data_factor"],
            )
            for item in record["datasets"]
        )
    )


def _protocol_signature(record: dict[str, Any]) -> tuple[Any, ...]:
    protocol = record["protocol"]
    return (
        protocol["evaluator_commit"],
        protocol["evaluator_sha256"],
        protocol["startup_compile_excluded"],
        tuple(protocol["seeds"]),
        tuple(protocol["pair_order"]),
        protocol["warmup_iterations"],
        protocol["measured_iterations"],
        protocol["rounds"],
        protocol["training_steps"],
        protocol["progress_scene_id"],
        protocol["progress_scope"],
    )


def validate_study(
    records: Sequence[dict[str, Any]], *, require_report_matrix: bool = True
) -> list[dict[str, Any]]:
    if not records:
        raise ValidationError("study contains no records")
    validated = [
        validate_record(record, f"records[{index}]")
        for index, record in enumerate(records)
    ]
    ordered = sorted(validated, key=lambda item: item["attempt_index"])
    experiment_ids = [item["experiment_id"] for item in ordered]
    attempt_indices = [item["attempt_index"] for item in ordered]
    if len(set(experiment_ids)) != len(experiment_ids):
        raise ValidationError("experiment_id values must be unique")
    if len(set(attempt_indices)) != len(attempt_indices):
        raise ValidationError("attempt_index values must be unique")
    study_ids = {item["study_id"] for item in ordered}
    if len(study_ids) != 1:
        raise ValidationError("all records must have one study_id")

    anchor = ordered[0]
    for record in ordered[1:]:
        if _dataset_signature(record) != _dataset_signature(anchor):
            _fail(record["experiment_id"], "dataset scene/hash/factor matrix changed")
        if _protocol_signature(record) != _protocol_signature(anchor):
            _fail(record["experiment_id"], "immutable evaluator protocol changed")
        if record["baseline"] != anchor["baseline"]:
            _fail(record["experiment_id"], "common baseline commit/label changed")
        if record["environment"] != anchor["environment"]:
            _fail(record["experiment_id"], "pinned execution environment changed")

    publication = [item for item in ordered if item["publication"]]
    if not require_report_matrix:
        return ordered
    if not publication:
        raise ValidationError("no publication records selected")
    baselines = [item for item in publication if item["comparison_kind"] == "baseline"]
    if len(baselines) != 1:
        raise ValidationError("publication matrix requires exactly one baseline record")
    environments = {canonical_json(item["environment"]) for item in publication}
    if len(environments) != 1:
        raise ValidationError("publication records use different pinned environments")
    scene_ids = {item["scene_id"] for item in anchor["datasets"]}
    expected = {(scene, scope) for scene in scene_ids for scope in SCOPES}
    for record in publication:
        present = {(item["scene_id"], item["scope"]) for item in record["measurements"]}
        missing = sorted(expected - present)
        if missing:
            formatted = ", ".join(f"{scene}/{scope}" for scene, scope in missing)
            _fail(
                record["experiment_id"], f"publication matrix incomplete: {formatted}"
            )
    sequential_ablation(publication, bootstrap_samples=200)
    return ordered


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of no values")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_seed(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def paired_summary(
    measurement: dict[str, Any], *, bootstrap_samples: int = 10_000, seed_key: str = ""
) -> dict[str, Any]:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    baseline = [float(value) for value in measurement["baseline_samples"]]
    candidate = [float(value) for value in measurement["candidate_samples"]]
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("paired samples must be non-empty and have equal length")
    baseline_median = statistics.median(baseline)
    candidate_median = statistics.median(candidate)
    speedup = baseline_median / candidate_median
    rng = random.Random(
        _bootstrap_seed(
            f"{REPORT_GENERATOR_VERSION}:{seed_key}:"
            + json.dumps(measurement, sort_keys=True, separators=(",", ":"))
        )
    )
    baseline_draws: list[float] = []
    candidate_draws: list[float] = []
    draws: list[float] = []
    for _ in range(bootstrap_samples):
        indices = [rng.randrange(len(baseline)) for _ in baseline]
        baseline_draw = statistics.median(baseline[index] for index in indices)
        candidate_draw = statistics.median(candidate[index] for index in indices)
        baseline_draws.append(baseline_draw)
        candidate_draws.append(candidate_draw)
        draws.append(baseline_draw / candidate_draw)
    return {
        "unit": measurement["unit"],
        "repeat_count": len(baseline),
        "baseline_median": baseline_median,
        "baseline_median_ci95": [
            _percentile(baseline_draws, 2.5),
            _percentile(baseline_draws, 97.5),
        ],
        "candidate_median": candidate_median,
        "candidate_median_ci95": [
            _percentile(candidate_draws, 2.5),
            _percentile(candidate_draws, 97.5),
        ],
        "speedup_x": speedup,
        "speedup_ci95_x": [_percentile(draws, 2.5), _percentile(draws, 97.5)],
        "duration_reduction_percent": 100.0 * (1.0 - 1.0 / speedup),
    }


def _measurement(record: dict[str, Any], scene: str, scope: str) -> dict[str, Any]:
    matches = [
        item
        for item in record["measurements"]
        if item["scene_id"] == scene and item["scope"] == scope
    ]
    if len(matches) != 1:
        raise ValidationError(
            f"{record['experiment_id']}: expected one measurement for {scene}/{scope}"
        )
    return matches[0]


def _geometric_mean(values: Iterable[float]) -> float:
    normalized = list(values)
    if not normalized or any(value <= 0.0 for value in normalized):
        raise ValueError("geometric mean requires positive values")
    return math.exp(sum(math.log(value) for value in normalized) / len(normalized))


def composite_summary(
    record: dict[str, Any],
    scene_ids: Sequence[str],
    scope: str,
    *,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    measurements = [_measurement(record, scene, scope) for scene in scene_ids]
    sample_counts = {len(item["baseline_samples"]) for item in measurements}
    sample_counts.update(len(item["candidate_samples"]) for item in measurements)
    if len(sample_counts) != 1:
        raise ValidationError(
            f"{record['experiment_id']}: composite inputs have unaligned repeats"
        )
    repeat_count = sample_counts.pop()
    point = _geometric_mean(
        statistics.median(item["baseline_samples"])
        / statistics.median(item["candidate_samples"])
        for item in measurements
    )
    seed_material = canonical_json(
        {
            "experiment_id": record["experiment_id"],
            "scenes": list(scene_ids),
            "scope": scope,
            "measurements": measurements,
        }
    )
    rng = random.Random(_bootstrap_seed(seed_material))
    draws: list[float] = []
    for _ in range(bootstrap_samples):
        indices = [rng.randrange(repeat_count) for _ in range(repeat_count)]
        scene_speedups = []
        for item in measurements:
            baseline = statistics.median(
                item["baseline_samples"][index] for index in indices
            )
            candidate = statistics.median(
                item["candidate_samples"][index] for index in indices
            )
            scene_speedups.append(baseline / candidate)
        draws.append(_geometric_mean(scene_speedups))
    return {
        "scene_ids": list(scene_ids),
        "scope": scope,
        "speedup_x": point,
        "speedup_ci95_x": [_percentile(draws, 2.5), _percentile(draws, 97.5)],
        "duration_reduction_percent": 100.0 * (1.0 - 1.0 / point),
    }


def promoted_frontier(
    records: Sequence[dict[str, Any]], *, bootstrap_samples: int = 10_000
) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda item: item["attempt_index"])
    frontier: float | None = None
    output = []
    for record in ordered:
        protocol = record["protocol"]
        summary = None
        if record["status"] != "crash":
            measurement = _measurement(
                record,
                protocol["progress_scene_id"],
                protocol["progress_scope"],
            )
            summary = paired_summary(
                measurement,
                bootstrap_samples=bootstrap_samples,
                seed_key=record["experiment_id"],
            )
        if record["promoted"] and summary is not None:
            frontier = max(frontier or 0.0, summary["speedup_x"])
        output.append(
            {
                "attempt_index": record["attempt_index"],
                "experiment_id": record["experiment_id"],
                "label": record["candidate"]["label"],
                "status": record["status"],
                "promoted": record["promoted"],
                "score": summary,
                "frontier_speedup_x": frontier,
            }
        )
    return output


def sequential_ablation(
    records: Sequence[dict[str, Any]], *, bootstrap_samples: int = 10_000
) -> dict[str, Any]:
    ablation_records = [item for item in records if "ablation" in item]
    if len(ablation_records) < 2:
        raise ValidationError(
            "report requires at least two sequential ablation checkpoints"
        )
    groups = {item["ablation"]["group_id"] for item in ablation_records}
    if len(groups) != 1:
        raise ValidationError("report accepts one ablation group at a time")
    methods = {item["ablation"]["method"] for item in ablation_records}
    if methods != {"historical_checkpoint"}:
        raise ValidationError(
            "waterfall currently requires historical_checkpoint ablation records"
        )
    ordered = sorted(ablation_records, key=lambda item: item["ablation"]["order"])
    orders = [item["ablation"]["order"] for item in ordered]
    if orders != list(range(len(orders))):
        raise ValidationError("ablation orders must be contiguous and start at zero")
    if ordered[0]["comparison_kind"] != "baseline":
        raise ValidationError("ablation order zero must be the baseline record")
    scene_ids = [item["scene_id"] for item in ordered[0]["datasets"]]
    previous_reduction = 0.0
    steps = []
    for record in ordered:
        summary = composite_summary(
            record,
            scene_ids,
            "forward_backward",
            bootstrap_samples=bootstrap_samples,
        )
        reduction = summary["duration_reduction_percent"]
        ci_speedup = summary["speedup_ci95_x"]
        reduction_ci = [
            100.0 * (1.0 - 1.0 / ci_speedup[0]),
            100.0 * (1.0 - 1.0 / ci_speedup[1]),
        ]
        steps.append(
            {
                "order": record["ablation"]["order"],
                "component": record["ablation"]["component"],
                "experiment_id": record["experiment_id"],
                "cumulative_reduction_percent": reduction,
                "cumulative_reduction_ci95_percent": reduction_ci,
                "contribution_percentage_points": reduction - previous_reduction,
            }
        )
        previous_reduction = reduction
    return {
        "group_id": ordered[0]["ablation"]["group_id"],
        "method": "historical_checkpoint",
        "scope": "forward_backward",
        "scene_ids": scene_ids,
        "steps": steps,
    }


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_data_hash(records: Sequence[dict[str, Any]]) -> str:
    ordered = sorted(records, key=lambda item: item["attempt_index"])
    return hashlib.sha256(canonical_json(ordered).encode("utf-8")).hexdigest()


def build_report_spec(
    records: Sequence[dict[str, Any]], *, bootstrap_samples: int = 10_000
) -> dict[str, Any]:
    ordered = validate_study(records, require_report_matrix=True)
    publication = [item for item in ordered if item["publication"]]
    scene_ids = [item["scene_id"] for item in publication[0]["datasets"]]
    per_scene = []
    operations = []
    training = []
    for record in publication:
        per_scene.append(
            {
                "experiment_id": record["experiment_id"],
                "label": record["candidate"]["label"],
                "values": [
                    {
                        "scene_id": scene,
                        **paired_summary(
                            _measurement(record, scene, "forward_backward"),
                            bootstrap_samples=bootstrap_samples,
                            seed_key=f"{record['experiment_id']}:{scene}:forward_backward",
                        ),
                    }
                    for scene in scene_ids
                ],
            }
        )
        operations.append(
            {
                "experiment_id": record["experiment_id"],
                "label": record["candidate"]["label"],
                "values": [
                    {
                        "scope": scope,
                        **composite_summary(
                            record,
                            scene_ids,
                            scope,
                            bootstrap_samples=bootstrap_samples,
                        ),
                    }
                    for scope in SCOPES[:3]
                ],
            }
        )
        training_values = []
        for scene in scene_ids:
            summary = paired_summary(
                _measurement(record, scene, "training"),
                bootstrap_samples=bootstrap_samples,
                seed_key=f"{record['experiment_id']}:{scene}:training",
            )
            steps = record["protocol"]["training_steps"]
            candidate_seconds = summary["candidate_median"]
            baseline_seconds = summary["baseline_median"]
            speedup_ci = summary["speedup_ci95_x"]
            training_values.append(
                {
                    "scene_id": scene,
                    **summary,
                    "candidate_steps_per_second": steps / candidate_seconds,
                    "candidate_steps_per_second_ci95": [
                        steps / summary["candidate_median_ci95"][1],
                        steps / summary["candidate_median_ci95"][0],
                    ],
                    "baseline_steps_per_second": steps / baseline_seconds,
                    "throughput_speedup_ci95_x": speedup_ci,
                }
            )
        training.append(
            {
                "experiment_id": record["experiment_id"],
                "label": record["candidate"]["label"],
                "values": training_values,
            }
        )
    return {
        "schema_version": 1,
        "generator": f"benchmarks.performance_report/{REPORT_GENERATOR_VERSION}",
        "study_id": ordered[0]["study_id"],
        "source_sha256": canonical_data_hash(ordered),
        "bootstrap_samples": bootstrap_samples,
        "baseline": ordered[0]["baseline"],
        "scene_ids": scene_ids,
        "scopes": list(SCOPES),
        "per_scene": per_scene,
        "operations": operations,
        "training": training,
        "progress": promoted_frontier(ordered, bootstrap_samples=bootstrap_samples),
        "ablation": sequential_ablation(
            publication, bootstrap_samples=bootstrap_samples
        ),
    }
