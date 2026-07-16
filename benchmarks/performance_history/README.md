# Performance-history data

This directory contains the tracked contract for the deferred multi-scene
performance study. It does not contain benchmark claims.

## Files

- `registry.tsv` is an append-only experiment event log. Existing rows are
  never edited or removed. A future result appends a `keep`, `discard`, or
  `crash` event with the same `experiment_id`; it does not replace its
  `planned` event.
- `result.schema.json` describes detailed JSON/JSONL result records consumed by
  `python -m benchmarks.performance_report`.
- `core.py` performs strict dependency-free validation, paired bootstrap
  aggregation, promoted-frontier calculation, and sequential history
  attribution.

## Registry contract

The TSV columns are:

| Column | Meaning |
| --- | --- |
| `event_id` | Unique append event ID. |
| `recorded_at_utc` | ISO-8601 UTC timestamp. |
| `experiment_id` | Stable experiment ID shared by planning and outcome events. |
| `commit` | Full candidate commit. |
| `baseline_commit` | Full common reference commit. |
| `status` | `planned`, `keep`, `discard`, or `crash`. |
| `sequence` | Intended historical/attempt order. |
| `description` | Tab- and newline-free experiment description. |
| `evidence` | Result JSONL path/hash, or planning provenance. |

Commits in the initial registry are proposed historical checkpoints inferred
from repository history. They are plans, not measurements. Before running,
verify ancestry, runtime diffs, buildability under the pinned image, and the
immutability of the reference backend across the chain.

## Result records

One result record represents one candidate attempt against the common baseline.
Each measurement keeps aligned `baseline_samples` and `candidate_samples`; the
reporter resamples the pair index jointly. Publication records require at least
five balanced repeat pairs and all four scopes for every declared scene.
Discarded attempts must still include the frozen primary evaluator measurement.
Crash records may contain no measurements.

The canonical tracked input is JSONL, one object per line, appended in attempt
order. JSON arrays and `{\"records\": [...]}` envelopes are accepted for data
exchange and tests. Validate without writing assets:

```bash
python3 -m benchmarks.performance_report results.jsonl --validate-only
```

Generate deterministic SVGs, a machine-readable report spec, and a Markdown
index only after the publication matrix is complete:

```bash
python3 -m benchmarks.performance_report results.jsonl \
  --output-dir benchmark-results/performance-history/report
```

Every SVG embeds the canonical validated-data SHA-256 and generator version.
SVG is the publication source. A PNG may be derived for platforms that require
it, but the renderer name/version and SVG hash must accompany that derivative.
