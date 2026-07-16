from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "benchmarks/kernel_benchmark.json"
DEFAULT_OUTPUT = ROOT / "docs/source/assets/kernel-benchmark.svg"
WIDTH = 1200
HEIGHT = 650


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _text(
    x: float,
    y: float,
    value: str,
    class_name: str,
    *,
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{_fmt(x, 1)}" y="{_fmt(y, 1)}" class="{class_name}" '
        f'text-anchor="{anchor}">{html.escape(value)}</text>'
    )


def load_data(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("kernel benchmark schema_version must be 1")
    if data.get("benchmark") != "isolated-raster-kernel-latency":
        raise ValueError("unexpected kernel benchmark name")
    measurements = data.get("measurements")
    if not isinstance(measurements, list) or len(measurements) != 2:
        raise ValueError("expected exactly forward and backward measurements")
    if [item.get("operation") for item in measurements] != ["Forward", "Backward"]:
        raise ValueError("measurements must be ordered Forward, Backward")

    for item in measurements:
        reference = float(item["reference_median_ms"])
        optimized = float(item["optimized_median_ms"])
        reduction = float(item["latency_reduction_percent"])
        expected = 100.0 * (1.0 - optimized / reference)
        if not all(
            math.isfinite(value) and value > 0 for value in (reference, optimized)
        ):
            raise ValueError("latencies must be finite and positive")
        if optimized >= reference:
            raise ValueError("optimized latency must be below reference latency")
        if not math.isclose(reduction, expected, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError("latency reduction disagrees with medians")
        for key in ("reference_ci95_ms", "optimized_ci95_ms"):
            low, high = (float(value) for value in item[key])
            if not (0 < low <= float(item[key.replace("_ci95", "_median")]) <= high):
                raise ValueError(f"invalid {key}")
        if int(item["sample_count_each"]) != 500:
            raise ValueError("README chart requires 500 samples per series")
    return data


def render_svg(data: dict[str, Any], source_sha256: str) -> str:
    measurements = data["measurements"]
    plot_left = 260.0
    plot_right = 1090.0
    plot_width = plot_right - plot_left
    axis_max = 4.0
    group_y = (240.0, 425.0)
    bar_height = 34.0
    bar_gap = 12.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
        'aria-labelledby="title description">',
        '<title id="title">Isolated raster kernel latency on RTX 5090</title>',
        (
            '<desc id="description">Horizontal bars compare gsplat 1.5.3 '
            "reference and ptxsplat SM120 median latency for isolated forward and "
            "backward raster stages. Lower latency is better.</desc>"
        ),
        f"<metadata>source_sha256={source_sha256}</metadata>",
        """<style>
        text { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing: 0; }
        .title { fill: #111827; font-size: 30px; font-weight: 700; }
        .subtitle { fill: #4b5563; font-size: 16px; }
        .axis { fill: #6b7280; font-size: 13px; }
        .group { fill: #111827; font-size: 20px; font-weight: 700; }
        .series { fill: #374151; font-size: 14px; font-weight: 600; }
        .value { fill: #111827; font-size: 14px; font-weight: 700; }
        .gain { fill: #065f46; font-size: 16px; font-weight: 700; }
        .note { fill: #4b5563; font-size: 13px; }
        </style>""",
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>',
        _text(60, 58, "Isolated raster kernel latency", "title"),
        _text(
            60,
            88,
            "RTX 5090 · 1920×1080 grid-7 garden · packed SH3 · lower is better",
            "subtitle",
        ),
    ]

    legend_y = 125
    parts.extend(
        [
            '<rect x="60" y="112" width="18" height="18" rx="3" fill="#9ca3af"/>',
            _text(88, legend_y + 1, "gsplat 1.5.3 reference", "series"),
            '<rect x="290" y="112" width="18" height="18" rx="3" fill="#0f766e"/>',
            _text(318, legend_y + 1, "ptxsplat SM120", "series"),
        ]
    )

    plot_top = 160.0
    plot_bottom = 510.0
    for tick in range(5):
        x = plot_left + plot_width * tick / 4
        parts.append(
            f'<line x1="{_fmt(x, 1)}" y1="{plot_top}" x2="{_fmt(x, 1)}" '
            f'y2="{plot_bottom}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(_text(x, 535, f"{tick} ms", "axis", anchor="middle"))

    for index, item in enumerate(measurements):
        center_y = group_y[index]
        reference_y = center_y - bar_height - bar_gap / 2
        optimized_y = center_y + bar_gap / 2
        reference = float(item["reference_median_ms"])
        optimized = float(item["optimized_median_ms"])
        reduction = float(item["latency_reduction_percent"])

        parts.append(_text(60, center_y - 30, item["operation"], "group"))
        parts.append(_text(60, reference_y + 23, "Reference", "series"))
        parts.append(_text(60, optimized_y + 23, "ptxsplat", "series"))

        for value, ci, y, color in (
            (reference, item["reference_ci95_ms"], reference_y, "#9ca3af"),
            (optimized, item["optimized_ci95_ms"], optimized_y, "#0f766e"),
        ):
            width = plot_width * value / axis_max
            parts.append(
                f'<rect x="{plot_left}" y="{_fmt(y, 1)}" width="{_fmt(width, 1)}" '
                f'height="{bar_height}" rx="4" fill="{color}"/>'
            )
            ci_low = plot_left + plot_width * float(ci[0]) / axis_max
            ci_high = plot_left + plot_width * float(ci[1]) / axis_max
            ci_y = y + bar_height / 2
            parts.extend(
                [
                    f'<line x1="{_fmt(ci_low, 1)}" y1="{_fmt(ci_y, 1)}" '
                    f'x2="{_fmt(ci_high, 1)}" y2="{_fmt(ci_y, 1)}" '
                    'stroke="#111827" stroke-width="2"/>',
                    f'<line x1="{_fmt(ci_low, 1)}" y1="{_fmt(ci_y - 5, 1)}" '
                    f'x2="{_fmt(ci_low, 1)}" y2="{_fmt(ci_y + 5, 1)}" '
                    'stroke="#111827" stroke-width="2"/>',
                    f'<line x1="{_fmt(ci_high, 1)}" y1="{_fmt(ci_y - 5, 1)}" '
                    f'x2="{_fmt(ci_high, 1)}" y2="{_fmt(ci_y + 5, 1)}" '
                    'stroke="#111827" stroke-width="2"/>',
                    _text(plot_left + width + 12, y + 23, f"{value:.3f} ms", "value"),
                ]
            )

        parts.append(
            _text(
                plot_right,
                center_y - 30,
                f"{reduction:.1f}% lower latency",
                "gain",
                anchor="end",
            )
        )

    parts.extend(
        [
            '<line x1="60" y1="565" x2="1140" y2="565" stroke="#d1d5db" stroke-width="1"/>',
            _text(
                60,
                592,
                "500 CUDA-event samples per series (20 warmups, 5 × 100); whiskers show bootstrap median 95% CI.",
                "note",
            ),
            _text(
                60,
                618,
                "Backward includes four required output zero-initializations; no projection, sorting, loss, optimizer, or training time.",
                "note",
            ),
            "</svg>",
            "",
        ]
    )
    return "\n".join(parts)


def generate(data_path: Path, output_path: Path) -> str:
    source_bytes = data_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    data = load_data(data_path)
    rendered = render_svg(data, source_sha256)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return source_sha256


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the README kernel benchmark SVG"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source_sha256 = generate(args.data, args.output)
    print(f"{args.output} (source sha256: {source_sha256})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
