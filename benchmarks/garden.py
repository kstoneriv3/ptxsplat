from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ptxsplat._helper import load_test_data
from ptxsplat.rendering import rasterization

from ._common import (
    cuda_event_samples,
    environment_metadata,
    parse_background,
    parse_csv,
    parse_resolution,
    require_cuda,
    summarize_samples,
    write_json,
)


SH_C0 = 0.28209479177387814
COLOR_MODES = ("rgb", "sh0", "sh1", "sh2", "sh3")
WORKLOADS = ("forward", "forward-backward")


@dataclass(frozen=True)
class Case:
    scene_grid: int
    width: int
    height: int
    color_mode: str
    background_name: str
    background_rgb: tuple[float, float, float] | None
    workload: str


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark packed single-camera garden rasterization with CUDA events."
    )
    parser.add_argument(
        "--scene-grids",
        default="1",
        help="Comma-separated odd scene grids (default: 1; full: 1,3,7).",
    )
    parser.add_argument(
        "--resolutions",
        default="180p",
        help="Comma-separated presets or WIDTHxHEIGHT values (default: 180p).",
    )
    parser.add_argument(
        "--color-modes",
        default="rgb",
        help="Comma-separated rgb/sh0/sh1/sh2/sh3 modes (default: rgb).",
    )
    parser.add_argument(
        "--backgrounds",
        default="black",
        help="Semicolon-separated none/black/white/gray/R,G,B values.",
    )
    parser.add_argument(
        "--workloads",
        default="forward,forward-backward",
        help="Comma-separated forward and forward-backward workloads.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--radius-clip", type=float, default=0.0)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets" / "test_garden.npz",
    )
    parser.add_argument("--output", default="-", help="JSON path, or - for stdout.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use the documented publication matrix and timing counts.",
    )
    return parser


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.full:
        args.scene_grids = "1,3,7"
        args.resolutions = "720p,1080p"
        args.color_modes = "sh3"
        args.backgrounds = "black"
        args.workloads = "forward,forward-backward"
        args.warmup = 20
        args.iterations = 100
        args.rounds = 5
    if args.warmup < 0 or args.iterations <= 0 or args.rounds <= 0:
        parser.error(
            "warmup must be nonnegative; iterations and rounds must be positive"
        )
    return args


def _parse_cases(args: argparse.Namespace) -> list[Case]:
    scene_grids = parse_csv(args.scene_grids, int)
    if any(grid <= 0 or grid % 2 == 0 for grid in scene_grids):
        raise ValueError("scene grids must be positive odd integers")
    resolutions = [parse_resolution(value) for value in parse_csv(args.resolutions)]
    color_modes = parse_csv(args.color_modes)
    if invalid := sorted(set(color_modes) - set(COLOR_MODES)):
        raise ValueError(f"unsupported color modes: {', '.join(invalid)}")
    workloads = parse_csv(args.workloads)
    if invalid := sorted(set(workloads) - set(WORKLOADS)):
        raise ValueError(f"unsupported workloads: {', '.join(invalid)}")
    backgrounds = [
        parse_background(value)
        for value in args.backgrounds.split(";")
        if value.strip()
    ]
    if not backgrounds:
        raise ValueError("expected at least one background")
    return [
        Case(grid, width, height, color, bg_name, bg_rgb, workload)
        for grid in scene_grids
        for width, height in resolutions
        for color in color_modes
        for bg_name, bg_rgb in backgrounds
        for workload in workloads
    ]


def _colors_for_mode(rgb: torch.Tensor, mode: str) -> tuple[torch.Tensor, int | None]:
    if mode == "rgb":
        return rgb.clone(), None
    degree = int(mode[-1])
    coefficients = torch.zeros(
        (rgb.shape[0], (degree + 1) ** 2, 3), device=rgb.device, dtype=rgb.dtype
    )
    coefficients[:, 0, :] = (rgb - 0.5) / SH_C0
    return coefficients, degree


def _run_case(case: Case, args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    (
        means,
        quats,
        scales,
        opacities,
        rgb,
        viewmats,
        intrinsics,
        fixture_width,
        fixture_height,
    ) = load_test_data(
        data_path=str(args.fixture), device="cuda", scene_grid=case.scene_grid
    )
    viewmats = viewmats[:1].contiguous()
    intrinsics = intrinsics[:1].clone()
    intrinsics[..., 0, :] *= case.width / fixture_width
    intrinsics[..., 1, :] *= case.height / fixture_height
    colors, sh_degree = _colors_for_mode(rgb, case.color_mode)
    background = (
        None
        if case.background_rgb is None
        else torch.tensor([case.background_rgb], device="cuda", dtype=torch.float32)
    )

    parameters = [means, quats, scales, opacities, colors]
    for parameter in parameters:
        parameter.requires_grad_(True)
    target = torch.full(
        (1, case.height, case.width, 3), 0.5, device="cuda", dtype=torch.float32
    )

    def render() -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        return rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            intrinsics,
            case.width,
            case.height,
            sh_degree=sh_degree,
            packed=True,
            backgrounds=background,
            render_mode="RGB",
            camera_model="pinhole",
            radius_clip=args.radius_clip,
        )

    if case.workload == "forward":
        operation = render
    else:

        def operation() -> torch.Tensor:
            rendered, _, _ = render()
            loss = F.mse_loss(rendered, target)
            loss.backward()
            return loss

    def clear_gradients() -> None:
        for parameter in parameters:
            parameter.grad = None

    with torch.no_grad():
        preflight_colors, preflight_alphas, preflight_meta = render()
        if not torch.isfinite(preflight_colors).all().item():
            raise RuntimeError("preflight render produced non-finite colors")
        if not torch.isfinite(preflight_alphas).all().item():
            raise RuntimeError("preflight render produced non-finite alphas")
        visible_gaussians = int(preflight_meta["radii"].numel())
        intersections = int(preflight_meta["flatten_ids"].numel())
        del preflight_colors, preflight_alphas, preflight_meta

    torch.cuda.empty_cache()
    baseline_memory = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    samples = cuda_event_samples(
        operation,
        warmup=args.warmup,
        iterations=args.iterations,
        rounds=args.rounds,
        after_iteration=clear_gradients,
    )
    timing = summarize_samples(
        samples,
        seed=args.seed + case.scene_grid + case.width + (sh_degree or 0),
    )
    peak_memory = torch.cuda.max_memory_allocated()
    return {
        "config": {
            "scene_grid": case.scene_grid,
            "resolution": [case.width, case.height],
            "color_mode": case.color_mode,
            "sh_degree": sh_degree,
            "background": case.background_name,
            "workload": case.workload,
            "packed": True,
            "camera_count": 1,
            "camera_model": "pinhole",
            "render_mode": "RGB",
            "radius_clip": args.radius_clip,
        },
        "input": {
            "gaussian_count": int(means.shape[0]),
            "visible_gaussian_count": visible_gaussians,
            "intersection_count": intersections,
            "fixture_resolution": [int(fixture_width), int(fixture_height)],
        },
        "timing": timing,
        "memory": {
            "baseline_allocated_bytes": int(baseline_memory),
            "peak_allocated_bytes": int(peak_memory),
            "peak_delta_bytes": int(peak_memory - baseline_memory),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        cases = _parse_cases(args)
        require_cuda()
        if not args.fixture.is_file():
            raise FileNotFoundError(f"garden fixture not found: {args.fixture}")
        results = []
        for index, case in enumerate(cases, start=1):
            print(
                f"[{index}/{len(cases)}] grid={case.scene_grid} "
                f"{case.width}x{case.height} {case.color_mode} "
                f"bg={case.background_name} {case.workload}",
                file=sys.stderr,
            )
            results.append(_run_case(case, args))
        write_json(
            {
                "schema_version": 1,
                "benchmark": "garden",
                "timing_method": "torch.cuda.Event",
                "seed": args.seed,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "rounds": args.rounds,
                "environment": environment_metadata(),
                "results": results,
            },
            args.output,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
