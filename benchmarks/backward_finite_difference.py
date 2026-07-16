"""Finite-difference audit for the low-level 3DGS raster backward path."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from benchmarks._common import environment_metadata, require_cuda, write_json
from ptxsplat.cuda._wrapper import rasterize_to_pixels


EPSILON_SWEEP = (0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625)
SMOOTH_ATOL = 5e-4
SMOOTH_RTOL = 3e-3
CLOSED_FORM_ATOL = 2e-6
CLOSED_FORM_RTOL = 2e-6
ALPHA_THRESHOLD = 1.0 / 255.0
ALPHA_CLAMP = 0.999
EARLY_TERMINATION_THRESHOLD = 1e-4
BACKENDS = ("reference", "sm120")


@dataclass(frozen=True)
class Scene:
    """One deterministic low-level raster fixture and its targeted Gaussian."""

    name: str
    build: Callable[[torch.device], dict[str, Any]]
    target_gaussian: int = 0


@dataclass(frozen=True)
class ScalarProbe:
    name: str
    parameter: str
    component: int


SCALAR_PROBES = (
    ScalarProbe("means_x", "means2d", 0),
    ScalarProbe("means_y", "means2d", 1),
    ScalarProbe("conic_xx", "conics", 0),
    ScalarProbe("conic_xy", "conics", 1),
    ScalarProbe("conic_yy", "conics", 2),
    ScalarProbe("rgb_0", "colors", 0),
    ScalarProbe("rgb_1", "colors", 1),
    ScalarProbe("rgb_2", "colors", 2),
    ScalarProbe("opacity", "opacities", 0),
)


def _offsets(device: torch.device) -> torch.Tensor:
    return torch.zeros((1, 1, 1), dtype=torch.int32, device=device)


def _single_off_center_scene(device: torch.device) -> dict[str, Any]:
    return {
        "means2d": torch.tensor([[[1.35, 2.10]]], device=device),
        "conics": torch.tensor([[[0.11, 0.015, 0.16]]], device=device),
        "colors": torch.tensor([[[0.21, 0.67, 0.38]]], device=device),
        "opacities": torch.tensor([[0.53]], device=device),
        "image_width": 5,
        "image_height": 4,
        "tile_size": 16,
        "isect_offsets": _offsets(device),
        "flatten_ids": torch.tensor([0], dtype=torch.int32, device=device),
        "packed": False,
    }


def _two_gaussian_scene(device: torch.device) -> dict[str, Any]:
    return {
        "means2d": torch.tensor(
            [[[1.60, 2.90], [4.25, 1.25]]], device=device
        ),
        "conics": torch.tensor(
            [[[0.09, 0.018, 0.12], [0.14, -0.012, 0.105]]], device=device
        ),
        "colors": torch.tensor(
            [[[0.75, 0.22, 0.11], [0.12, 0.42, 0.84]]], device=device
        ),
        "opacities": torch.tensor([[0.37, 0.44]], device=device),
        "image_width": 6,
        "image_height": 5,
        "tile_size": 16,
        "isect_offsets": _offsets(device),
        "flatten_ids": torch.tensor([0, 1], dtype=torch.int32, device=device),
        "packed": False,
    }


SCENES = (
    Scene("single_off_center", _single_off_center_scene),
    Scene("two_gaussian_composite", _two_gaussian_scene),
)


def _parameter_index(scene: Scene, probe: ScalarProbe) -> tuple[int, ...]:
    if probe.parameter == "opacities":
        return (0, scene.target_gaussian)
    return (0, scene.target_gaussian, probe.component)


def _parameter_label(parameter: str, index: tuple[int, ...]) -> str:
    return f"{parameter}[{','.join(str(part) for part in index)}]"


def _loss(inputs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    render, alpha = rasterize_to_pixels(**inputs)
    height, width = alpha.shape[-3:-1]
    pixel_index = torch.arange(height * width, dtype=render.dtype, device=render.device)
    pixel_weight = (0.11 + 0.017 * pixel_index).reshape(1, height, width, 1)
    color_weight = torch.tensor([0.31, -0.47, 0.83], device=render.device)
    alpha_weight = (0.19 + 0.013 * pixel_index).reshape(1, height, width, 1)
    loss = (render * pixel_weight * color_weight).sum()
    loss = loss + (alpha * alpha_weight).sum()
    return loss, render, alpha


def _set_backend(backend: str) -> str | None:
    previous = os.environ.get("PTXSPLAT_BACKEND")
    os.environ["PTXSPLAT_BACKEND"] = backend
    return previous


def _restore_backend(previous: str | None) -> None:
    if previous is None:
        os.environ.pop("PTXSPLAT_BACKEND", None)
    else:
        os.environ["PTXSPLAT_BACKEND"] = previous


def _evaluate_parameter(
    backend: str,
    scene: Scene,
    device: torch.device,
    probe: ScalarProbe,
    delta: float = 0.0,
    requires_grad: bool = False,
) -> tuple[float, float | None, bool, float]:
    previous_backend = _set_backend(backend)
    try:
        inputs = scene.build(device)
        index = _parameter_index(scene, probe)
        tensor = inputs[probe.parameter]
        tensor[index] += delta
        tensor.requires_grad_(requires_grad)
        loss, render, alpha = _loss(inputs)
        gradient = None
        if requires_grad:
            loss.backward()
            gradient = float(tensor.grad[index].item())
        torch.cuda.synchronize(device)
        finite = bool(torch.isfinite(render).all() and torch.isfinite(alpha).all())
        return float(loss.item()), gradient, finite, float(alpha.sum().item())
    finally:
        _restore_backend(previous_backend)


def _select_stable_difference(sweep: list[dict[str, Any]]) -> tuple[int, float]:
    """Select a central difference without looking at an analytic gradient."""

    pairs = [
        (index, abs(sweep[index]["numerical"] - sweep[index + 1]["numerical"]))
        for index in range(len(sweep) - 1)
        if sweep[index]["finite"] and sweep[index + 1]["finite"]
    ]
    if not pairs:
        raise RuntimeError("finite-difference epsilon sweep contains no finite pair")
    pair_index, disagreement = min(pairs, key=lambda item: item[1])
    return pair_index + 1, disagreement


def _run_probe(
    backend: str, scene: Scene, device: torch.device, probe: ScalarProbe
) -> dict[str, Any]:
    index = _parameter_index(scene, probe)
    base_inputs = scene.build(device)
    parameter_value = float(base_inputs[probe.parameter][index].item())
    _, analytic, finite, alpha_sum = _evaluate_parameter(
        backend, scene, device, probe, requires_grad=True
    )
    assert analytic is not None

    sweep: list[dict[str, Any]] = []
    for epsilon in EPSILON_SWEEP:
        plus, _, plus_finite, _ = _evaluate_parameter(
            backend, scene, device, probe, delta=epsilon
        )
        minus, _, minus_finite, _ = _evaluate_parameter(
            backend, scene, device, probe, delta=-epsilon
        )
        sweep.append(
            {
                "epsilon": epsilon,
                "loss_plus": plus,
                "loss_minus": minus,
                "numerical": (plus - minus) / (2.0 * epsilon),
                "finite": plus_finite and minus_finite,
            }
        )

    selected_index, stability = _select_stable_difference(sweep)
    selected = sweep[selected_index]
    numerical = float(selected["numerical"])
    absolute_error = abs(analytic - numerical)
    scale = max(abs(analytic), abs(numerical))
    limit = SMOOTH_ATOL + SMOOTH_RTOL * scale
    return {
        "backend": backend,
        "scene": scene.name,
        "probe": probe.name,
        "parameter": _parameter_label(probe.parameter, index),
        "parameter_group": probe.parameter,
        "parameter_index": list(index),
        "parameter_value": parameter_value,
        "analytic_gradient": analytic,
        "numerical_gradient": numerical,
        "absolute_error": absolute_error,
        "relative_error": absolute_error / max(scale, 1e-12),
        "epsilon": selected["epsilon"],
        "epsilon_selection_index": selected_index,
        "neighbor_disagreement": stability,
        "epsilon_sweep": sweep,
        "forward_finite": finite,
        "active_alpha_sum": alpha_sum,
        "error_limit": limit,
        "passed": finite and alpha_sum > 0.0 and absolute_error <= limit,
    }


def _fp64_closed_form(inputs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    means = inputs["means2d"].detach().cpu().double()[0]
    conics = inputs["conics"].detach().cpu().double()[0]
    colors = inputs["colors"].detach().cpu().double()[0]
    opacities = inputs["opacities"].detach().cpu().double()[0]
    ids = inputs["flatten_ids"].detach().cpu().tolist()
    height = int(inputs["image_height"])
    width = int(inputs["image_width"])
    expected_render = torch.zeros((1, height, width, 3), dtype=torch.float64)
    expected_alpha = torch.zeros((1, height, width, 1), dtype=torch.float64)
    min_sigma = math.inf
    min_alpha_margin = math.inf
    min_clamp_margin = math.inf
    min_termination_margin = math.inf

    for y in range(height):
        for x in range(width):
            transmittance = 1.0
            color = torch.zeros(3, dtype=torch.float64)
            for gaussian_id in ids:
                delta_x = float(means[gaussian_id, 0]) - (x + 0.5)
                delta_y = float(means[gaussian_id, 1]) - (y + 0.5)
                sigma = (
                    0.5
                    * (
                        float(conics[gaussian_id, 0]) * delta_x * delta_x
                        + float(conics[gaussian_id, 2]) * delta_y * delta_y
                    )
                    + float(conics[gaussian_id, 1]) * delta_x * delta_y
                )
                raw_alpha = float(opacities[gaussian_id]) * math.exp(-sigma)
                min_sigma = min(min_sigma, sigma)
                min_alpha_margin = min(min_alpha_margin, raw_alpha - ALPHA_THRESHOLD)
                min_clamp_margin = min(min_clamp_margin, ALPHA_CLAMP - raw_alpha)
                if sigma < 0.0 or raw_alpha < ALPHA_THRESHOLD:
                    continue
                alpha = min(ALPHA_CLAMP, raw_alpha)
                next_transmittance = transmittance * (1.0 - alpha)
                min_termination_margin = min(
                    min_termination_margin,
                    next_transmittance - EARLY_TERMINATION_THRESHOLD,
                )
                if next_transmittance <= EARLY_TERMINATION_THRESHOLD:
                    break
                color += colors[gaussian_id] * (alpha * transmittance)
                transmittance = next_transmittance
            expected_render[0, y, x] = color
            expected_alpha[0, y, x, 0] = 1.0 - transmittance

    return expected_render, expected_alpha, {
        "min_sigma": min_sigma,
        "min_alpha_threshold_margin": min_alpha_margin,
        "min_alpha_clamp_margin": min_clamp_margin,
        "min_early_termination_margin": min_termination_margin,
    }


def _closed_form_check(backend: str, scene: Scene, device: torch.device) -> dict[str, Any]:
    previous_backend = _set_backend(backend)
    try:
        inputs = scene.build(device)
        _, render, alpha = _loss(inputs)
        torch.cuda.synchronize(device)
        expected_render, expected_alpha, margins = _fp64_closed_form(inputs)
        actual_render = render.detach().cpu().double()
        actual_alpha = alpha.detach().cpu().double()
        color_error = float((actual_render - expected_render).abs().max().item())
        opacity_error = float((actual_alpha - expected_alpha).abs().max().item())
        scale = max(
            float(actual_render.abs().max().item()),
            float(expected_render.abs().max().item()),
            float(actual_alpha.abs().max().item()),
            float(expected_alpha.abs().max().item()),
        )
        limit = CLOSED_FORM_ATOL + CLOSED_FORM_RTOL * scale
        smooth_interior = all(value > 0.0 for value in margins.values())
        finite = bool(torch.isfinite(render).all() and torch.isfinite(alpha).all())
        return {
            "backend": backend,
            "scene": scene.name,
            "oracle": "fp64_front_to_back_compositing",
            "color_max_absolute_error": color_error,
            "opacity_max_absolute_error": opacity_error,
            "error_limit": limit,
            "forward_finite": finite,
            "smooth_interior": smooth_interior,
            "smooth_margins": margins,
            "passed": finite
            and smooth_interior
            and color_error <= limit
            and opacity_error <= limit,
        }
    finally:
        _restore_backend(previous_backend)


def _nonsmooth_classifications(closed_form: list[dict[str, Any]]) -> list[dict[str, Any]]:
    minima = {
        "min_alpha_threshold_margin": min(
            check["smooth_margins"]["min_alpha_threshold_margin"]
            for check in closed_form
        ),
        "min_alpha_clamp_margin": min(
            check["smooth_margins"]["min_alpha_clamp_margin"]
            for check in closed_form
        ),
        "min_sigma": min(
            check["smooth_margins"]["min_sigma"] for check in closed_form
        ),
        "min_early_termination_margin": min(
            check["smooth_margins"]["min_early_termination_margin"]
            for check in closed_form
        ),
    }
    return [
        {
            "name": "alpha_threshold",
            "trigger": {"alpha": ALPHA_THRESHOLD},
            "classification": "nonsmooth_excluded",
            "finite_difference_policy": "exclude the exact boundary; central differences use only the active interior",
            "smooth_interior_min_margin": minima["min_alpha_threshold_margin"],
        },
        {
            "name": "alpha_clamp_0.999",
            "trigger": {"raw_alpha": ALPHA_CLAMP},
            "classification": "nonsmooth_excluded",
            "finite_difference_policy": "exclude the clamp boundary; clamped-interior geometry and opacity gradients are separately expected to be zero",
            "smooth_interior_min_margin": minima["min_alpha_clamp_margin"],
        },
        {
            "name": "negative_sigma",
            "trigger": {"sigma": 0.0},
            "classification": "nonsmooth_excluded",
            "finite_difference_policy": "exclude the sigma sign boundary; smooth probes require sigma strictly positive",
            "smooth_interior_min_margin": minima["min_sigma"],
        },
        {
            "name": "early_termination",
            "trigger": {"next_transmittance": EARLY_TERMINATION_THRESHOLD},
            "classification": "nonsmooth_excluded",
            "finite_difference_policy": "exclude the exclusive termination boundary; smooth probes retain a positive transmittance margin",
            "smooth_interior_min_margin": minima["min_early_termination_margin"],
        },
        {
            "name": "tile_membership_and_last_id",
            "trigger": {"flatten_id_order_or_last_id": "changes"},
            "classification": "discrete_excluded",
            "finite_difference_policy": "low-level intersections are fixed; perturbations do not regenerate tile membership and must not cross a last-id change",
            "fixed_intersections": True,
            "scenes": [scene.name for scene in SCENES],
        },
    ]


def _error_summaries(probes: list[dict[str, Any]]) -> dict[str, Any]:
    by_backend: dict[str, dict[str, Any]] = {}
    by_parameter: dict[str, dict[str, Any]] = {}
    for key, grouping in (("backend", by_backend), ("parameter_group", by_parameter)):
        for value in sorted({probe[key] for probe in probes}):
            matching = [probe for probe in probes if probe[key] == value]
            grouping[value] = {
                "count": len(matching),
                "failed": sum(not probe["passed"] for probe in matching),
                "max_absolute_error": max(probe["absolute_error"] for probe in matching),
                "max_relative_error": max(probe["relative_error"] for probe in matching),
            }
    return {
        "probe_count": len(probes),
        "failed_probe_count": sum(not probe["passed"] for probe in probes),
        "max_absolute_error": max(probe["absolute_error"] for probe in probes),
        "max_relative_error": max(probe["relative_error"] for probe in probes),
        "by_backend": by_backend,
        "by_parameter_group": by_parameter,
    }


def run_audit(device: torch.device) -> dict[str, Any]:
    closed_form = [
        _closed_form_check(backend, scene, device)
        for backend in BACKENDS
        for scene in SCENES
    ]
    probes = [
        _run_probe(backend, scene, device, probe)
        for backend in BACKENDS
        for scene in SCENES
        for probe in SCALAR_PROBES
    ]
    summaries = _error_summaries(probes)
    classifications = _nonsmooth_classifications(closed_form)
    passed = (
        len(probes) >= 36
        and all(probe["passed"] for probe in probes)
        and all(check["passed"] for check in closed_form)
        and all(
            classification.get("smooth_interior_min_margin", 1.0) > 0.0
            for classification in classifications
        )
    )
    return {
        "schema_version": 1,
        "audit": "backward-finite-difference",
        "backends": list(BACKENDS),
        "scenes": [scene.name for scene in SCENES],
        "expected_probe_count": len(BACKENDS) * len(SCENES) * len(SCALAR_PROBES),
        "criteria": {"atol": SMOOTH_ATOL, "rtol": SMOOTH_RTOL},
        "epsilon_sweep": list(EPSILON_SWEEP),
        "epsilon_selection": "smaller epsilon from the adjacent finite central-difference pair with minimum numerical disagreement; analytic gradients are not consulted",
        "closed_form_criteria": {"atol": CLOSED_FORM_ATOL, "rtol": CLOSED_FORM_RTOL},
        "orchestration": {
            "reasoning_effort": "xhigh",
            "execution_model": "Terra",
            "selection_reason": "deliberate fallback after two Sol execution workers stalled",
            "stalled_sol_workers": 2,
        },
        "environment": environment_metadata(),
        "closed_form_checks": closed_form,
        "nonsmooth_classifications": classifications,
        "probes": probes,
        "error_summaries": summaries,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/backward-finite-difference/analysis.json"),
    )
    args = parser.parse_args()
    require_cuda()
    result = run_audit(torch.device("cuda:0"))
    write_json(result, str(args.output))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
