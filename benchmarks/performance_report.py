from __future__ import annotations

import argparse
import html
import json
import math
import sys
import textwrap
from pathlib import Path
from typing import Any, Iterable

from .performance_history import (
    REPORT_GENERATOR_VERSION,
    ValidationError,
    build_report_spec,
    canonical_data_hash,
    load_records,
    validate_study,
)

WIDTH = 1200
PALETTE = ("#16697a", "#d95f39", "#6a994e", "#7b5ea7", "#cc8b00", "#3a506b")
STATUS_COLORS = {"keep": "#2f855a", "discard": "#d97706", "crash": "#c53030"}
GRID = "#d7dde2"
TEXT = "#17212b"
MUTED = "#5e6b75"
BACKGROUND = "#ffffff"


def _fmt(value: float, digits: int = 2) -> str:
    rendered = f"{value:.{digits}f}"
    return rendered.rstrip("0").rstrip(".")


class Svg:
    def __init__(self, height: int, title: str, metadata: dict[str, Any]) -> None:
        self.height = height
        self.parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
                f'height="{height}" viewBox="0 0 {WIDTH} {height}" role="img" '
                f'aria-label="{html.escape(title, quote=True)}" '
                f'data-source-sha256="{metadata["source_sha256"]}">'
            ),
            f"<title>{html.escape(title)}</title>",
            f"<metadata>{html.escape(json.dumps(metadata, sort_keys=True))}</metadata>",
            (
                "<style>text{font-family:Inter,Arial,sans-serif;letter-spacing:0}"
                ".title{font-size:24px;font-weight:700;fill:#17212b}"
                ".subtitle{font-size:13px;fill:#5e6b75}"
                ".axis{font-size:12px;fill:#5e6b75}"
                ".label{font-size:12px;fill:#17212b}"
                ".value{font-size:11px;font-weight:600;fill:#17212b}"
                ".legend{font-size:12px;fill:#17212b}</style>"
            ),
            f'<rect width="{WIDTH}" height="{height}" fill="{BACKGROUND}"/>',
        ]

    def add(self, value: str) -> None:
        self.parts.append(value)

    def text(
        self,
        x: float,
        y: float,
        value: str,
        css: str = "label",
        *,
        anchor: str = "start",
        transform: str | None = None,
    ) -> None:
        transform_attr = f' transform="{transform}"' if transform else ""
        self.add(
            f'<text x="{_fmt(x)}" y="{_fmt(y)}" class="{css}" '
            f'text-anchor="{anchor}"{transform_attr}>{html.escape(value)}</text>'
        )

    def finish(self) -> str:
        return "\n".join([*self.parts, "</svg>", ""])


def _metadata(spec: dict[str, Any], chart: str) -> dict[str, Any]:
    return {
        "chart": chart,
        "generator": f"benchmarks.performance_report/{REPORT_GENERATOR_VERSION}",
        "schema_version": spec["schema_version"],
        "source_sha256": spec["source_sha256"],
        "study_id": spec["study_id"],
        "baseline_commit": spec["baseline"]["commit"],
        "bootstrap_samples": spec["bootstrap_samples"],
    }


def _header(svg: Svg, title: str, subtitle: str) -> None:
    svg.text(64, 45, title, "title")
    svg.text(64, 69, subtitle, "subtitle")


def _legend(svg: Svg, entries: list[dict[str, Any]], y: int = 94) -> None:
    x = 64
    for index, entry in enumerate(entries):
        label = entry["label"]
        width = max(120, 20 + len(label) * 7)
        if x + width > WIDTH - 64:
            y += 23
            x = 64
        color = PALETTE[index % len(PALETTE)]
        svg.add(f'<rect x="{x}" y="{y - 11}" width="13" height="13" fill="{color}"/>')
        svg.text(x + 19, y, label, "legend")
        x += width


def _nice_max(value: float) -> float:
    if value <= 0.0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    normalized = value / magnitude
    step = (
        1.0
        if normalized <= 1.0
        else 2.0
        if normalized <= 2.0
        else 5.0
        if normalized <= 5.0
        else 10.0
    )
    return step * magnitude


def _grouped_speedup_svg(
    spec: dict[str, Any],
    *,
    chart: str,
    title: str,
    subtitle: str,
    entries: list[dict[str, Any]],
    categories: list[str],
    category_key: str,
) -> str:
    height = 690
    svg = Svg(height, title, _metadata(spec, chart))
    _header(svg, title, subtitle)
    _legend(svg, entries)
    left, right, top, bottom = 80, WIDTH - 45, 135, height - 110
    plot_width, plot_height = right - left, bottom - top
    highs = [
        value["speedup_ci95_x"][1] for entry in entries for value in entry["values"]
    ]
    y_max = max(1.1, _nice_max(max(highs) * 1.08))
    for tick in range(6):
        value = y_max * tick / 5
        y = bottom - plot_height * value / y_max
        svg.add(
            f'<line x1="{left}" y1="{_fmt(y)}" x2="{right}" y2="{_fmt(y)}" stroke="{GRID}"/>'
        )
        svg.text(left - 10, y + 4, f"{_fmt(value)}x", "axis", anchor="end")
    baseline_y = bottom - plot_height / y_max
    svg.add(
        f'<line x1="{left}" y1="{_fmt(baseline_y)}" x2="{right}" '
        f'y2="{_fmt(baseline_y)}" stroke="#4b5563" stroke-width="1.5" stroke-dasharray="5 4"/>'
    )
    svg.text(right, baseline_y - 7, "1.0x reference", "axis", anchor="end")
    group_width = plot_width / len(categories)
    bar_gap = 4
    usable = group_width * 0.8
    bar_width = min(54.0, (usable - bar_gap * (len(entries) - 1)) / len(entries))
    for category_index, category in enumerate(categories):
        center = left + group_width * (category_index + 0.5)
        total = len(entries) * bar_width + (len(entries) - 1) * bar_gap
        start = center - total / 2
        for entry_index, entry in enumerate(entries):
            value = next(
                item for item in entry["values"] if item[category_key] == category
            )
            speedup = value["speedup_x"]
            ci_low, ci_high = value["speedup_ci95_x"]
            x = start + entry_index * (bar_width + bar_gap)
            y = bottom - plot_height * speedup / y_max
            color = PALETTE[entry_index % len(PALETTE)]
            svg.add(
                f'<rect x="{_fmt(x)}" y="{_fmt(y)}" width="{_fmt(bar_width)}" '
                f'height="{_fmt(bottom - y)}" fill="{color}" opacity="0.9"/>'
            )
            error_x = x + bar_width / 2
            low_y = bottom - plot_height * ci_low / y_max
            high_y = bottom - plot_height * ci_high / y_max
            svg.add(
                f'<path d="M {_fmt(error_x)} {_fmt(low_y)} V {_fmt(high_y)} '
                f"M {_fmt(error_x - 5)} {_fmt(low_y)} H {_fmt(error_x + 5)} "
                f'M {_fmt(error_x - 5)} {_fmt(high_y)} H {_fmt(error_x + 5)}" '
                'stroke="#17212b" stroke-width="1.2" fill="none"/>'
            )
        svg.text(
            center, bottom + 25, category.replace("_", " "), "label", anchor="middle"
        )
    svg.text(
        20,
        (top + bottom) / 2,
        "Speedup (baseline duration / candidate duration)",
        "axis",
        anchor="middle",
        transform=f"rotate(-90 20 {_fmt((top + bottom) / 2)})",
    )
    svg.text(
        WIDTH - 45,
        height - 24,
        f"Data SHA-256: {spec['source_sha256']}",
        "subtitle",
        anchor="end",
    )
    return svg.finish()


def _training_svg(spec: dict[str, Any]) -> str:
    title = "End-to-end training time and throughput"
    height = 840
    svg = Svg(height, title, _metadata(spec, "training_time_throughput"))
    _header(
        svg,
        title,
        "Fixed 7,000-step training-loop scope; lower time and higher throughput are better. Bars are paired medians.",
    )
    entries = spec["training"]
    _legend(svg, entries)
    scenes = spec["scene_ids"]
    panels = [
        (135, 420, "Training time (minutes)", "time"),
        (495, 780, "Training throughput (steps/second)", "throughput"),
    ]
    for top, bottom, label, mode in panels:
        left, right = 80, WIDTH - 45
        values = []
        for entry in entries:
            for item in entry["values"]:
                if mode == "time":
                    values.append(item["candidate_median_ci95"][1] / 60.0)
                else:
                    values.append(item["candidate_steps_per_second_ci95"][1])
        y_max = _nice_max(max(values) * 1.1)
        plot_height = bottom - top
        for tick in range(5):
            value = y_max * tick / 4
            y = bottom - plot_height * value / y_max
            svg.add(
                f'<line x1="{left}" y1="{_fmt(y)}" x2="{right}" y2="{_fmt(y)}" stroke="{GRID}"/>'
            )
            svg.text(left - 10, y + 4, _fmt(value), "axis", anchor="end")
        svg.text(left, top - 13, label, "label")
        group_width = (right - left) / len(scenes)
        bar_width = min(54.0, group_width * 0.75 / len(entries))
        for scene_index, scene in enumerate(scenes):
            center = left + group_width * (scene_index + 0.5)
            start = center - bar_width * len(entries) / 2
            for entry_index, entry in enumerate(entries):
                item = next(
                    value for value in entry["values"] if value["scene_id"] == scene
                )
                if mode == "time":
                    value = item["candidate_median"] / 60.0
                    ci_low, ci_high = (
                        item["candidate_median_ci95"][0] / 60.0,
                        item["candidate_median_ci95"][1] / 60.0,
                    )
                else:
                    value = item["candidate_steps_per_second"]
                    ci_low, ci_high = item["candidate_steps_per_second_ci95"]
                x = start + entry_index * bar_width
                y = bottom - plot_height * value / y_max
                svg.add(
                    f'<rect x="{_fmt(x)}" y="{_fmt(y)}" width="{_fmt(bar_width - 3)}" '
                    f'height="{_fmt(bottom - y)}" fill="{PALETTE[entry_index % len(PALETTE)]}" opacity="0.9"/>'
                )
                error_x = x + (bar_width - 3) / 2
                low_y = bottom - plot_height * ci_low / y_max
                high_y = bottom - plot_height * ci_high / y_max
                svg.add(
                    f'<path d="M {_fmt(error_x)} {_fmt(low_y)} V {_fmt(high_y)} '
                    f"M {_fmt(error_x - 5)} {_fmt(low_y)} H {_fmt(error_x + 5)} "
                    f'M {_fmt(error_x - 5)} {_fmt(high_y)} H {_fmt(error_x + 5)}" '
                    'stroke="#17212b" stroke-width="1.2" fill="none"/>'
                )
            svg.text(center, bottom + 22, scene, "label", anchor="middle")
    svg.text(
        WIDTH - 45,
        height - 24,
        f"Data SHA-256: {spec['source_sha256']}",
        "subtitle",
        anchor="end",
    )
    return svg.finish()


def _progress_svg(spec: dict[str, Any]) -> str:
    title = "Optimization progress and promoted frontier"
    height = 680
    svg = Svg(height, title, _metadata(spec, "optimization_progress"))
    progress = spec["progress"]
    first_score = next(item["score"] for item in progress if item["score"] is not None)
    _header(
        svg,
        title,
        "Every attempt uses the immutable primary evaluator; crash attempts remain in the outcome strip.",
    )
    left, right, top, bottom = 85, WIDTH - 50, 115, 520
    scored = [item for item in progress if item["score"] is not None]
    all_y = [item["score"]["speedup_ci95_x"][0] for item in scored]
    all_y.extend(item["score"]["speedup_ci95_x"][1] for item in scored)
    all_y.extend(
        item["frontier_speedup_x"]
        for item in progress
        if item["frontier_speedup_x"] is not None
    )
    y_min = min(0.95, min(all_y) * 0.98)
    y_max = max(1.05, max(all_y) * 1.02)
    if y_max - y_min < 0.05:
        y_max = y_min + 0.05
    x_min = min(item["attempt_index"] for item in progress)
    x_max = max(item["attempt_index"] for item in progress)
    x_span = max(1, x_max - x_min)

    def x_pos(value: int) -> float:
        return left + (right - left) * (value - x_min) / x_span

    def y_pos(value: float) -> float:
        return bottom - (bottom - top) * (value - y_min) / (y_max - y_min)

    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = y_pos(value)
        svg.add(
            f'<line x1="{left}" y1="{_fmt(y)}" x2="{right}" y2="{_fmt(y)}" stroke="{GRID}"/>'
        )
        svg.text(left - 10, y + 4, f"{_fmt(value, 3)}x", "axis", anchor="end")
    frontier_points = [
        (x_pos(item["attempt_index"]), y_pos(item["frontier_speedup_x"]))
        for item in progress
        if item["frontier_speedup_x"] is not None
    ]
    if frontier_points:
        path = " ".join(
            ("M" if index == 0 else "L") + f" {_fmt(x)} {_fmt(y)}"
            for index, (x, y) in enumerate(frontier_points)
        )
        svg.add(f'<path d="{path}" fill="none" stroke="#2f855a" stroke-width="3"/>')
    for item in progress:
        x = x_pos(item["attempt_index"])
        if item["score"] is not None:
            score = item["score"]
            y = y_pos(score["speedup_x"])
            low = y_pos(score["speedup_ci95_x"][0])
            high = y_pos(score["speedup_ci95_x"][1])
            color = STATUS_COLORS[item["status"]]
            svg.add(
                f'<line x1="{_fmt(x)}" y1="{_fmt(low)}" x2="{_fmt(x)}" y2="{_fmt(high)}" stroke="{color}"/>'
            )
            svg.add(
                f'<circle cx="{_fmt(x)}" cy="{_fmt(y)}" r="5" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>'
            )
        lane_y = 570
        color = STATUS_COLORS[item["status"]]
        if item["status"] == "crash":
            svg.add(
                f'<path d="M {_fmt(x - 5)} {lane_y - 5} L {_fmt(x + 5)} {lane_y + 5} M {_fmt(x - 5)} {lane_y + 5} L {_fmt(x + 5)} {lane_y - 5}" stroke="{color}" stroke-width="2"/>'
            )
        else:
            svg.add(f'<circle cx="{_fmt(x)}" cy="{lane_y}" r="5" fill="{color}"/>')
        svg.text(x, 596, str(item["attempt_index"]), "axis", anchor="middle")
    svg.text(left, 548, "Outcome strip", "label")
    legend_x = 790
    for index, status in enumerate(("keep", "discard", "crash")):
        x = legend_x + index * 105
        svg.add(f'<circle cx="{x}" cy="92" r="5" fill="{STATUS_COLORS[status]}"/>')
        svg.text(x + 10, 96, status, "legend")
    svg.add(
        '<line x1="650" y1="92" x2="690" y2="92" stroke="#2f855a" stroke-width="3"/>'
    )
    svg.text(700, 96, "promoted frontier", "legend")
    svg.text(
        20,
        (top + bottom) / 2,
        "Primary evaluator speedup",
        "axis",
        anchor="middle",
        transform=f"rotate(-90 20 {_fmt((top + bottom) / 2)})",
    )
    svg.text(
        WIDTH - 50,
        height - 24,
        f"Data SHA-256: {spec['source_sha256']}",
        "subtitle",
        anchor="end",
    )
    _ = first_score
    return svg.finish()


def _waterfall_svg(spec: dict[str, Any]) -> str:
    title = "Sequential optimization-history contribution"
    height = 760
    svg = Svg(height, title, _metadata(spec, "sequential_history_waterfall"))
    ablation = spec["ablation"]
    _header(
        svg,
        title,
        "Cumulative historical checkpoints; contribution is order-dependent and is not an independent causal ablation.",
    )
    steps = ablation["steps"]
    left, right, top, bottom = 85, WIDTH - 50, 125, 570
    all_values = [0.0]
    for step in steps:
        all_values.extend(step["cumulative_reduction_ci95_percent"])
        all_values.append(step["cumulative_reduction_percent"])
    y_min = min(0.0, min(all_values))
    y_max = max(1.0, max(all_values))
    padding = max(1.0, (y_max - y_min) * 0.15)
    y_min -= padding
    y_max += padding

    def y_pos(value: float) -> float:
        return bottom - (bottom - top) * (value - y_min) / (y_max - y_min)

    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = y_pos(value)
        svg.add(
            f'<line x1="{left}" y1="{_fmt(y)}" x2="{right}" y2="{_fmt(y)}" stroke="{GRID}"/>'
        )
        svg.text(left - 10, y + 4, f"{_fmt(value)}%", "axis", anchor="end")
    group_width = (right - left) / len(steps)
    bar_width = min(100.0, group_width * 0.58)
    previous = 0.0
    for index, step in enumerate(steps):
        center = left + group_width * (index + 0.5)
        current = step["cumulative_reduction_percent"]
        low, high = sorted((previous, current))
        color = "#2f855a" if current >= previous else "#c53030"
        svg.add(
            f'<rect x="{_fmt(center - bar_width / 2)}" y="{_fmt(y_pos(high))}" '
            f'width="{_fmt(bar_width)}" height="{_fmt(max(2.0, y_pos(low) - y_pos(high)))}" fill="{color}" opacity="0.88"/>'
        )
        if index < len(steps) - 1:
            next_center = left + group_width * (index + 1.5)
            svg.add(
                f'<line x1="{_fmt(center + bar_width / 2)}" y1="{_fmt(y_pos(current))}" x2="{_fmt(next_center - bar_width / 2)}" y2="{_fmt(y_pos(current))}" stroke="#6b7280" stroke-dasharray="4 3"/>'
            )
        ci_low, ci_high = step["cumulative_reduction_ci95_percent"]
        svg.add(
            f'<line x1="{_fmt(center)}" y1="{_fmt(y_pos(ci_low))}" x2="{_fmt(center)}" y2="{_fmt(y_pos(ci_high))}" stroke="#17212b" stroke-width="1.2"/>'
        )
        svg.text(
            center,
            y_pos(current) - 8,
            f"{step['contribution_percentage_points']:+.2f} pp",
            "value",
            anchor="middle",
        )
        lines = textwrap.wrap(step["component"], width=18)[:3]
        for line_index, line in enumerate(lines):
            svg.text(
                center, bottom + 28 + line_index * 16, line, "label", anchor="middle"
            )
        previous = current
    svg.text(
        20,
        (top + bottom) / 2,
        "Cumulative duration reduction",
        "axis",
        anchor="middle",
        transform=f"rotate(-90 20 {_fmt((top + bottom) / 2)})",
    )
    svg.text(
        WIDTH - 50,
        height - 24,
        f"Data SHA-256: {spec['source_sha256']}",
        "subtitle",
        anchor="end",
    )
    return svg.finish()


def render_assets(spec: dict[str, Any]) -> dict[str, str]:
    operation_labels = {
        "isolated_forward": "isolated forward",
        "isolated_backward": "isolated backward",
        "forward_backward": "forward + backward",
    }
    operation_entries = []
    for entry in spec["operations"]:
        operation_entries.append(
            {
                **entry,
                "values": [
                    {**value, "scope_label": operation_labels[value["scope"]]}
                    for value in entry["values"]
                ],
            }
        )
    return {
        "per-scene-speedup.svg": _grouped_speedup_svg(
            spec,
            chart="per_scene_speedup",
            title="Per-scene forward + backward speedup",
            subtitle=f"Paired medians and 95% bootstrap intervals versus {spec['baseline']['label']}.",
            entries=spec["per_scene"],
            categories=spec["scene_ids"],
            category_key="scene_id",
        ),
        "operation-breakdown.svg": _grouped_speedup_svg(
            spec,
            chart="operation_breakdown",
            title="Isolated and combined rendering speedup",
            subtitle="Geometric mean across the frozen scene matrix; forward + backward is measured directly.",
            entries=operation_entries,
            categories=list(operation_labels.values()),
            category_key="scope_label",
        ),
        "training-time-throughput.svg": _training_svg(spec),
        "optimization-progress.svg": _progress_svg(spec),
        "sequential-history-waterfall.svg": _waterfall_svg(spec),
    }


def _report_markdown(spec: dict[str, Any], assets: Iterable[str]) -> str:
    lines = [
        f"# Performance report: {spec['study_id']}",
        "",
        f"Baseline: `{spec['baseline']['label']}` (`{spec['baseline']['commit']}`).",
        "",
        f"Validated data SHA-256: `{spec['source_sha256']}`.",
        "",
        "The sequential waterfall is order-dependent historical attribution, not an independent causal ablation.",
        "",
    ]
    for asset in assets:
        label = asset.removesuffix(".svg").replace("-", " ").title()
        lines.extend([f"## {label}", "", f"![{label}]({asset})", ""])
    return "\n".join(lines)


def write_report(spec: dict[str, Any], output_dir: Path) -> None:
    assets = render_assets(spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in assets.items():
        (output_dir / name).write_text(content, encoding="utf-8")
    encoded_spec = json.dumps(spec, indent=2, sort_keys=True, allow_nan=False) + "\n"
    (output_dir / "report-spec.json").write_text(encoded_spec, encoding="utf-8")
    (output_dir / "REPORT.md").write_text(
        _report_markdown(spec, assets), encoding="utf-8"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate paired performance-history results and generate deterministic SVG reports."
    )
    parser.add_argument("inputs", nargs="+", help="Result JSON or JSONL paths.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark-results/performance-history/report"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.bootstrap_samples <= 0:
        print("error: --bootstrap-samples must be positive", file=sys.stderr)
        return 2
    try:
        records = load_records(args.inputs)
        validate_study(records, require_report_matrix=not args.validate_only)
        if args.validate_only:
            print(canonical_data_hash(records))
            return 0
        spec = build_report_spec(records, bootstrap_samples=args.bootstrap_samples)
        write_report(spec, args.output_dir)
    except (OSError, ValidationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
