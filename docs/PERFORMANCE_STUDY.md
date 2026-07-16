# Reproducible Performance-History Study

This is the executable protocol and publication gate for a deferred performance
study. It defines no performance result. GPU runs must wait until Bonsai and
other viewers are stopped and the device is exclusively available.

The design adapts the fixed evaluator, fixed evaluation budget, and append-only
`keep`/`discard`/`crash` experiment log from
[Karpathy's autoresearch](https://github.com/karpathy/autoresearch). Here the
immutable score is paired rendering/training duration under a fixed scene and
step matrix, not model validation loss.

## Study questions

The study has three distinct questions. Do not combine their labels:

1. **Current impact:** current optimized `auto` versus the gsplat 1.5.3
   reference backend over multiple scenes.
2. **Optimization history:** cumulative historical checkpoints versus one
   common reference. A sequential waterfall is order-dependent attribution,
   not proof of an independent causal effect.
3. **Causal ablation:** at the final commit, remove one promoted mechanism at a
   time while holding every other source and build input fixed. Publish this
   separately; interactions mean one-at-a-time effects need not sum to the
   historical total.

## Frozen scene matrix

Use the official Mip-NeRF 360 `360_v2` dataset and the existing trainer's
canonical downsampling convention:

| Scene | Setting | Factor | Purpose |
| --- | --- | ---: | --- |
| `bonsai` | indoor | 2 | Primary scene; dense foliage and overlap. |
| `room` | indoor | 2 | Larger surfaces and a less foliage-dominated indoor case. |
| `garden` | outdoor | 4 | Existing project oracle and dense outdoor vegetation. |
| `bicycle` | outdoor | 4 | Broad outdoor views, fine geometry, and high coverage. |

This spans indoor/outdoor capture and different expected overlap patterns.
Do not assert density ordering from scene names. The evaluator must record
per-step Gaussian count, visible count, intersections, and pixel-Gaussian
evaluations; the report may describe density only from those measurements.

Download through `examples/datasets/download_dataset.py`. Before any build or
run, create and retain a relative-path file manifest for each extracted scene:

```bash
cd data/360_v2/bonsai
find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > manifest.sha256
sha256sum manifest.sha256
```

Repeat for all four scenes. The outer SHA-256 enters every result record. The
manifest itself is a study artifact. Any changed file, path, factor, parser, or
train/test split creates a new study ID.

## Immutable evaluator and environment

Before the first measurement, commit the evaluator/collector and record both
its commit and a SHA-256 over its source/config files. The evaluator is then
read-only for the whole study. A correction starts a new study; old records
remain append-only. It must call the public gsplat-compatible API and must not
copy candidate implementation code.

Pin and record:

- full candidate and common-baseline commits, with a clean worktree;
- `360-video-gs-dev:latest` resolved to its immutable image ID/digest;
- GPU UUID/name (GPU 0, RTX 5090, SM120), driver, CUDA, PyTorch, Python, nvcc,
  compile flags, and built wheel SHA-256;
- evaluator commit/hash, dataset manifests, trainer config, seeds, and exact
  command line;
- locked graphics/memory clocks and power limit when the host permits them.

`latest` alone is not a publication identity. Do not mix image IDs, drivers,
GPU UUIDs, build flags, or clock policies in one publication report.

Build only through `scripts/docker-run.sh`. Create detached historical
worktrees under the ignored result tree so each commit has isolated source and
build outputs:

```bash
git worktree add --detach \
  benchmark-results/performance-history/worktrees/a821317 \
  a821317e30a5a24acb1cb8db6f56837d9973ceac
./scripts/docker-run.sh -- bash -lc \
  'cd benchmark-results/performance-history/worktrees/a821317 && \
   python3 -m build --wheel --outdir ../../wheels/a821317'
sha256sum benchmark-results/performance-history/wheels/a821317/*.whl
```

Run the frozen evaluator against one candidate wheel per fresh container or
isolated virtual environment. Never import an extension from another
worktree's build/JIT cache. Archive build logs and dispatch proof, but do not
commit wheels, extensions, cubins, SASS, or Nsight reports.

## Historical checkpoints

The planned append-only registry is
`benchmarks/performance_history/registry.tsv`. Its initial cumulative chain is:

| Order | Commit | Planned checkpoint |
| ---: | --- | --- |
| 0 | `4d6b94d` | Pre-SM120 reference-only checkpoint. |
| 10 | `a821317` | Initial promoted SM120 backward. |
| 20 | `fc88cbd` | Initial promoted SM120 forward plus backward. |
| 30 | `41b462c` | Reused-stage output transpose. |
| 40 | `9b7df04` | Stage-major backward reductions. |
| 50 | `a4973d5` | Compact 8x4 backward mapping. |

Before execution, verify ancestry and diff every adjacent pair. Commits between
these points include benchmark/correctness changes. Exclude a checkpoint if its
runtime diff is not attributable or build-compatible; append a `discard` or
`crash` event explaining why rather than rewriting the plan.

## Fixed budgets and scopes

### Training

Use `examples/simple_trainer.py default` with `packed=true`, batch size 1,
DefaultStrategy, SH degree 3, one GPU, viewer disabled, fixed scene factor, and
exactly 7,000 optimization steps. Disable evaluation, rendering, TensorBoard
image writes, PLY export, and intermediate checkpoint writes during timing.
Use seeds `42, 43, 44, 45, 46` for five paired repeats.

The timed interval starts after process startup, dataset parsing, worker
creation/priming, allocations, extension loading/JIT compilation, and a
sacrificial compile preflight. Reset model, optimizer, strategy, RNG, and data
order after preflight. Synchronize immediately before step 0 and after step
6,999. Include data-loader stalls, host-to-device copies, projection, sorting,
raster forward, loss, raster backward, optimizer, scheduler, and densification.
This is **end-to-end training-loop time**, not command wall time. Record command
wall time separately as a diagnostic. Report seconds and derived steps/second.

Record final quality (PSNR/SSIM/LPIPS), loss, peak memory, and Gaussian count.
Timing promotion is invalid if the candidate changes the fixed quality gate or
trains a materially different workload. A fixed step budget compares throughput;
it does not claim equal time-to-quality.

### Isolated rendering

Generate one frozen step-6,999 checkpoint and a deterministic camera list per
scene with the reference evaluator. Every candidate consumes the same checkpoint
and camera order.

- **Isolated forward:** CUDA events enclose the complete public rasterization
  call, including projection/intersection/sort/offset/SH/raster stages. Do not
  include checkpoint load, allocation, or compile.
- **Isolated backward:** prepare the exact saved forward state outside the timed
  interval; events enclose gradient-output setup required by the public backward
  path, output zeroing performed by that path, and backward execution. Restore
  fresh gradients/state before each sample.
- **Forward+backward:** events enclose the public forward, fixed MSE loss, and
  backward together. It is measured directly, never reconstructed by addition.

For publication use 20 warmups, 100 event-timed iterations, five rounds, and
five paired process-level repeats. Keep eager launch overhead. Record a CUDA
graph replay only as a separately named diagnostic.

## Pairing, clocks, and thermal control

Confirm no viewer, training process, profiler, or other GPU client is active.
Warm the GPU with the frozen warmup workload. If clock locking is available,
set and record one fixed supported graphics/memory clock pair and power limit.
If not, retain round-bracketing telemetry and reject a pair when either run:

- starts above 85 C;
- starts more than 3 C from its mate;
- has median SM or memory clock more than 1% from its mate; or
- overlaps another GPU process, ECC/Xid event, throttling transition, or
  profiler activity.

Alternate execution order by repeat:

`baseline,candidate`; `candidate,baseline`; `baseline,candidate`;
`candidate,baseline`; `baseline,candidate`.

Use a fresh process for each side. Pair runs from the same block and resample
the pair index jointly. Failed/rejected blocks remain logged and are rerun as a
new append event; do not silently delete samples.

## Statistics and correctness gate

The reference backend and existing tests are the oracle. Before timing each
build, run the applicable repository correctness suite, finite-difference
audit, deterministic frozen-camera parity checks, and dispatch proof. Preserve
current tolerances and adversarial cases. A crash, fallback when optimized
dispatch was required, NaN/Inf, quality-gate failure, or missing provenance is
not publishable.

For each scene/scope report paired medians, IQR in the raw artifact, baseline
and candidate medians, speedup `baseline_duration / candidate_duration`, and a
10,000-resample paired bootstrap 95% confidence interval. The multi-scene score
is the geometric mean of per-scene speedups with repeat blocks resampled
jointly. Retain all raw samples. Do not average per-scene percentages or pool
iterations as independent process repeats.

The fixed progress evaluator is Bonsai forward+backward. Every non-crash
attempt, including discarded attempts, records it. The promoted frontier moves
only on a correctness-passing `keep` marked as promoted. Crashes remain visible
in the outcome strip.

## Comparability hazards

The study report must state these hazards and any mitigation actually used:

- Historical commits can change dispatch, wrappers, correctness behavior, or
  reference code in addition to the named kernel; inspect adjacent runtime
  diffs and hash reference sources.
- `auto` can silently fall back. Capture selected backend/kernel for every
  measured shape.
- Same-source A/B variant toggles are stronger causal evidence than separate
  historical builds; do not label cumulative commit deltas as independent.
- Different compiler, flags, extension/JIT cache, image, driver, clocks, or
  wheel contamination can dominate small gains.
- Dynamic densification amplifies floating-point differences into different
  Gaussian counts and work. Report trajectory statistics and quality, and use
  frozen checkpoints for isolated scopes.
- Data-loader order/workers, camera sampling, background RNG, and asynchronous
  timing can break pairing unless reset from the recorded seeds.
- Indoor factor 2 and outdoor factor 4 intentionally differ. Compare a scene
  only with its own baseline; use geometric means for the summary.
- Forward+backward is not forward plus an independently timed backward.
- Live Bonsai viewers consume compute/memory and invalidate all pairs.
- Multiple checkpoint/scene comparisons increase false discovery risk. Treat
  confidence intervals as uncertainty, not proof of universality.

## Append-only outcomes and report generation

Do not edit prior rows in `registry.tsv`. Append one outcome event keyed by full
commit/experiment ID with `keep`, `discard`, or `crash`, a concise description,
and the detailed result JSONL path plus SHA-256. Detailed records conform to
`benchmarks/performance_history/result.schema.json` and retain paired samples.

Generate assets with:

```bash
python3 -m benchmarks.performance_report \
  benchmark-results/performance-history/results.jsonl \
  --output-dir benchmark-results/performance-history/report \
  --bootstrap-samples 10000
```

The command fails on an incomplete scene/scope matrix and writes deterministic
SVGs for per-scene speedup, scope breakdown, training time/throughput, all-attempt
progress with promoted frontier, and sequential history contribution. It also
writes `report-spec.json` and `REPORT.md`. Every asset embeds the validated-data
hash and explicit baseline.

## Publication gate

README performance charts or claims are forbidden until all of the following
are true:

- all four scenes and all four scopes have five valid balanced pairs;
- dataset, evaluator, candidate wheel, environment, and raw result hashes are
  archived and the schema/report validation passes;
- correctness, quality, dispatch, clock, thermal, and fixed-work checks pass;
- charts show uncertainty and name the common baseline;
- historical waterfall is labeled order-dependent, and any causal ablation is
  separately identified;
- a reviewer can reproduce the report from result JSONL with the documented
  command.

Only then may verified SVGs be copied to a tracked documentation asset path and
embedded in `README.md` with the study ID, hardware, matrix, and result link.
Placeholder charts and single-scene headline claims are never embedded.

## First dynamic batch to queue later

After viewers are stopped and the collector is committed, queue one calibration
batch before historical worktrees:

1. Current optimized commit versus forced reference in fresh isolated builds.
2. Bonsai only, factor 2, packed DefaultStrategy, batch 1, SH degree 3.
3. Seeds `42..46`, balanced order `AB/BA/AB/BA/AB`, 7,000 training steps.
4. Frozen step-6,999 checkpoint for isolated forward, isolated backward, and
   direct forward+backward at native factor-2 image sizes.
5. Rendering: 20 warmups, 100 iterations, five rounds per repeat; paired
   bootstrap with 10,000 resamples.
6. Run the full correctness/quality/dispatch and thermal gates, append the
   outcome, and generate a **local calibration report only**.

If that batch is stable and complete, queue the remaining three scenes for the
same two builds, then historical checkpoints in balanced blocks. This ordering
finds evaluator or thermal defects before spending time on the history matrix.
