from __future__ import annotations

import math

import pytest

from benchmarks.kernel_ceiling import (
    criteria_from_ranges,
    dynamic_subopcode_counts,
    independent_resource_bound_ms,
    normalize_ncu_metric,
    parse_ncu_csv,
    parse_number,
    parse_opcode_mix,
    parse_pc_instances,
    parse_sass_functions,
    reduction_algorithmic_counts,
    select_sass_function,
)


def test_parse_number_accepts_ncu_formatting() -> None:
    assert parse_number("1,234,567") == 1_234_567
    assert parse_number("22.5%") == 22.5
    assert math.isnan(parse_number("n/a"))


def test_parse_ncu_wide_csv_with_units() -> None:
    text = """==PROF== ignored
\"ID\",\"Kernel Name\",\"gpu__time_duration.sum\"
\"\",\"\",\"nsecond\"
\"0\",\"kernel(float *)\",\"1,250.0\"
"""
    rows, units = parse_ncu_csv(text)
    assert rows == [
        {
            "ID": "0",
            "Kernel Name": "kernel(float *)",
            "gpu__time_duration.sum": "1,250.0",
        }
    ]
    assert units["gpu__time_duration.sum"] == "nsecond"


def test_normalize_archived_ncu_csv_regression_values() -> None:
    text = """\
\"ID\",\"Kernel Name\",\"gpu__time_duration.sum\",\"lts__t_bytes.sum\"
\"\",\"\",\"ms\",\"Mbyte\"
\"0\",\"kernel(float *)\",\"1.626176\",\"481.489824\"
"""
    rows, units = parse_ncu_csv(text)
    duration = normalize_ncu_metric(
        rows[0]["gpu__time_duration.sum"],
        units["gpu__time_duration.sum"],
        "time",
    )
    traffic = normalize_ncu_metric(
        rows[0]["lts__t_bytes.sum"], units["lts__t_bytes.sum"], "bytes"
    )

    assert duration.original_value == "1.626176"
    assert duration.original_unit == "ms"
    assert duration.normalized_value == 1.626176
    assert duration.normalized_unit == "ms"
    assert traffic.original_value == "481.489824"
    assert traffic.original_unit == "Mbyte"
    assert traffic.normalized_value == pytest.approx(481_489_824.0)
    assert traffic.normalized_unit == "byte"


@pytest.mark.parametrize(
    ("unit", "expected_ms"),
    [
        ("ns", 1e-6),
        ("nsecond", 1e-6),
        ("us", 1e-3),
        ("usecond", 1e-3),
        ("ms", 1.0),
        ("msecond", 1.0),
        ("s", 1e3),
        ("second", 1e3),
    ],
)
def test_normalize_ncu_time_units(unit: str, expected_ms: float) -> None:
    metric = normalize_ncu_metric("1", unit, "time")
    assert metric.normalized_value == expected_ms
    assert metric.normalized_unit == "ms"


@pytest.mark.parametrize(
    ("unit", "expected_bytes"),
    [
        ("byte", 1.0),
        ("Kbyte", 1_000.0),
        ("Mbyte", 1_000_000.0),
        ("Gbyte", 1_000_000_000.0),
        ("KiB", 1_024.0),
        ("MiB", 1_048_576.0),
        ("GiB", 1_073_741_824.0),
    ],
)
def test_normalize_ncu_decimal_and_binary_byte_units(
    unit: str, expected_bytes: float
) -> None:
    metric = normalize_ncu_metric("1", unit, "bytes")
    assert metric.normalized_value == expected_bytes
    assert metric.normalized_unit == "byte"


def test_normalize_ncu_byte_per_block_unit() -> None:
    metric = normalize_ncu_metric("12.288", "Kbyte/block", "bytes")
    assert metric.normalized_value == 12_288.0
    assert metric.normalized_unit == "byte/block"


@pytest.mark.parametrize(
    ("unit", "dimension"),
    [("cycles", "time"), ("Kibit", "bytes"), ("Mbyte/s", "bytes")],
)
def test_normalize_ncu_rejects_unknown_units_loudly(unit: str, dimension: str) -> None:
    with pytest.raises(ValueError, match="unsupported NCU"):
        normalize_ncu_metric("1", unit, dimension)  # type: ignore[arg-type]


def test_parse_ncu_tall_csv_without_units() -> None:
    text = """\"ID\",\"Kernel Name\",\"Metric Name\",\"Metric Value\"
\"0\",\"kernel\",\"Executed Instructions\",\"20\"
"""
    rows, units = parse_ncu_csv(text)
    assert rows[0]["Metric Value"] == "20"
    assert units == {}


def test_parse_instruction_instances_and_opcode_mix() -> None:
    assert parse_opcode_mix("100 (SHFL: 55; FADD: 44; REDG: 11)") == {
        "SHFL": 55,
        "FADD": 44,
        "REDG": 11,
    }
    assert parse_pc_instances("30 (0x1000: 10; 0x1010: 20)") == {0x1000: 10, 0x1010: 20}


def test_sass_parse_select_and_dynamic_subopcode_mapping() -> None:
    sass = """
    Function : exact_kernel
        /*0000*/                   MUFU.EX2 R1, R2;
        /*0010*/                   MUFU.RCP R3, R4;
        /*0020*/                   BAR.SYNC.DEFER_BLOCKING 0x0;
    Function : unrelated
        /*0000*/                   FADD R1, R2, R3;
    """
    functions = parse_sass_functions(sass)
    name, instructions = select_sass_function(functions, "exact_kernel")
    assert name == "exact_kernel"
    assert [row["opcode"] for row in instructions] == [
        "MUFU.EX2",
        "MUFU.RCP",
        "BAR.SYNC.DEFER_BLOCKING",
    ]
    assert dynamic_subopcode_counts(
        {0x8000: 7, 0x8010: 5, 0x8020: 2}, instructions
    ) == {
        "MUFU.EX2": 7,
        "MUFU.RCP": 5,
        "BAR.SYNC.DEFER_BLOCKING": 2,
    }


def test_select_sass_function_rejects_ambiguous_match() -> None:
    with pytest.raises(ValueError, match="expected one"):
        select_sass_function({"foo_a": [], "foo_b": []}, "foo")


def test_exact_backward_reduction_and_atomic_minimum_counts() -> None:
    launched_warps = 65_280
    events = 12_345
    dynamic_shuffles = events * 9 * 5
    counts = reduction_algorithmic_counts(
        dynamic_shuffles=dynamic_shuffles,
        launched_warps=launched_warps,
    )
    assert counts["warp_active_gaussian_events"] == events
    assert counts["sum_shuffle_add_pairs"] == events * 45
    assert counts["minimum_redg_fp32_atomics"] == events * 9


def test_reduction_count_rejects_non_integral_event_count() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        reduction_algorithmic_counts(
            dynamic_shuffles=1,
            launched_warps=8,
            reductions_per_event=11,
        )


def test_independent_resources_use_max_not_sum() -> None:
    model = independent_resource_bound_ms({"fp32": 0.1, "l2": 0.4, "barrier": 0.3})
    assert model["lower_bound_ms"] == 0.4
    assert model["limiting_resource"] == "l2"
    assert model["equation"] == "max(fp32, l2, barrier)"


def test_sensitivity_criteria_require_worst_case() -> None:
    result = criteria_from_ranges(
        resource_target_low_ms=2.5,
        current_q95_ms=10.0,
        current_q05_ms=9.0,
        resource_target_high_ms=9.0,
    )
    assert result["at_least_25_percent_of_ceiling_established"]
    assert not result["less_than_10_percent_residual_established"]
    assert result["robust_minimum_efficiency_percent"] == 25.0
    assert result["residual_relative_to_resource_target_percent_range"] == [
        0.0,
        300.0,
    ]


def test_sensitivity_residual_can_pass_only_at_robust_edge() -> None:
    result = criteria_from_ranges(
        resource_target_low_ms=10.0,
        current_q95_ms=10.9,
        current_q05_ms=10.1,
        resource_target_high_ms=10.2,
    )
    assert result["at_least_25_percent_of_ceiling_established"]
    assert result["less_than_10_percent_residual_established"]
