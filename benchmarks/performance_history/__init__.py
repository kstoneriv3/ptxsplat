"""Validation and aggregation for reproducible performance-history reports."""

from .core import (
    REPORT_GENERATOR_VERSION,
    ValidationError,
    build_report_spec,
    canonical_data_hash,
    load_records,
    paired_summary,
    promoted_frontier,
    sequential_ablation,
    validate_study,
)

__all__ = [
    "REPORT_GENERATOR_VERSION",
    "ValidationError",
    "build_report_spec",
    "canonical_data_hash",
    "load_records",
    "paired_summary",
    "promoted_frontier",
    "sequential_ablation",
    "validate_study",
]
