from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from ptxsplat import __gsplat_version__, __version__


def parse_csv(value: str, convert: Callable[[str], Any] = str) -> list[Any]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("expected at least one comma-separated value")
    return [convert(item) for item in items]


def parse_resolution(value: str) -> tuple[int, int]:
    presets = {
        "180p": (320, 180),
        "360p": (640, 360),
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "4k": (3840, 2160),
    }
    normalized = value.strip().lower()
    if normalized in presets:
        return presets[normalized]
    try:
        width, height = (int(part) for part in normalized.split("x", maxsplit=1))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid resolution {value!r}; use a preset or WIDTHxHEIGHT"
        ) from exc
    if width <= 0 or height <= 0:
        raise ValueError("resolution dimensions must be positive")
    return width, height


def parse_background(value: str) -> tuple[str, tuple[float, float, float] | None]:
    normalized = value.strip().lower()
    named = {
        "none": None,
        "black": (0.0, 0.0, 0.0),
        "white": (1.0, 1.0, 1.0),
        "gray": (0.5, 0.5, 0.5),
    }
    if normalized in named:
        return normalized, named[normalized]
    try:
        rgb = tuple(float(part) for part in normalized.split(","))
    except ValueError as exc:
        raise ValueError(
            "background must be none, black, white, gray, or R,G,B"
        ) from exc
    if len(rgb) != 3 or any(channel < 0.0 or channel > 1.0 for channel in rgb):
        raise ValueError("background R,G,B channels must be in [0, 1]")
    return value, rgb  # type: ignore[return-value]


def cuda_event_samples(
    operation: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
    rounds: int,
    after_iteration: Callable[[], None] | None = None,
) -> list[list[float]]:
    for _ in range(warmup):
        result = operation()
        del result
        if after_iteration is not None:
            after_iteration()
    torch.cuda.synchronize()

    samples: list[list[float]] = []
    for _ in range(rounds):
        round_samples = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = operation()
            end.record()
            end.synchronize()
            round_samples.append(float(start.elapsed_time(end)))
            del result
            if after_iteration is not None:
                after_iteration()
        samples.append(round_samples)
    return samples


def summarize_samples(samples: list[list[float]], seed: int) -> dict[str, Any]:
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("cannot summarize an empty sample set")
    q1, median, q3 = np.percentile(values, [25.0, 50.0, 75.0])
    rng = np.random.default_rng(seed)
    bootstrap_medians = np.median(
        rng.choice(values, size=(2000, values.size), replace=True), axis=1
    )
    ci_low, ci_high = np.percentile(bootstrap_medians, [2.5, 97.5])
    return {
        "unit": "ms",
        "count": int(values.size),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "bootstrap_median_ci95": [float(ci_low), float(ci_high)],
        "round_medians": [float(np.median(round_values)) for round_values in samples],
        "samples": samples,
    }


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command, stderr=subprocess.DEVNULL, text=True, timeout=5
        ).strip()
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return None


def environment_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ptxsplat": __version__,
        "gsplat_compatibility": __gsplat_version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_command_output(["git", "status", "--porcelain"])),
        "container_hostname": platform.node(),
        "container_image_ref": os.environ.get(
            "PTXSPLAT_DOCKER_IMAGE", "360-video-gs-dev:latest"
        ),
        "driver_version": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
                "--id=0",
            ]
        ),
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        metadata["gpu"] = {
            "index": index,
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
        }
        query = _command_output(
            [
                "nvidia-smi",
                "--query-gpu=clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
                "--id=0",
            ]
        )
        if query:
            sm_clock, memory_clock, temperature, power = (
                item.strip() for item in query.split(",")
            )
            metadata["gpu"].update(
                {
                    "sm_clock_mhz": float(sm_clock),
                    "memory_clock_mhz": float(memory_clock),
                    "temperature_c": float(temperature),
                    "power_w": float(power),
                }
            )
    return metadata


def write_json(payload: dict[str, Any], output: str) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output == "-":
        sys.stdout.write(encoded)
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)
    print(path, file=sys.stderr)


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run through scripts/docker-run.sh")
