# PTX authoring on RTX 5090

This directory supports profile-driven PTX experiments without coupling the
runtime backend to an authoring tool. The initial smoke path is deliberately
small enough to inspect from CUDA source through final SASS.

## Toolchain contract

- Target GPU: RTX 5090, compute capability 12.0 (`sm_120`).
- Target toolkit: CUDA 12.8 or newer. CUDA 12.8 supplies PTX ISA 8.7 and the
  `sm_120` assembler target used by this repository.
- Assembly target: always pass `sm_120` explicitly to `ptxas`. Do not rely on
  the current device or toolkit defaults.
- Generated PTX, cubins, and SASS listings belong under `.bcodex/ptx/` and are
  not source artifacts.

Workstation Blackwell is not datacenter Blackwell or Hopper. Do not port an
`sm_100a` or `sm_90a` kernel by changing only its target directive. RTX 5090
does not provide the Hopper WGMMA path or the datacenter Blackwell `tcgen05`,
TMEM, and TMA path. Every instruction used by an `sm_120` experiment must
assemble with the repository's pinned target. Start from ordinary scalar,
vector, shared-memory, `mma.sync`, and `cp.async` building blocks only after
confirming each one with `ptxas` and a device correctness test.

## Smoke pipeline

Run the complete non-PyPTX pipeline inside the project container:

```bash
./scripts/docker-run.sh -- ./scripts/ptx-smoke.sh
```

This performs these independently callable stages:

```bash
./scripts/docker-run.sh -- ./scripts/ptx-codegen.sh
./scripts/docker-run.sh -- ./scripts/ptx-assemble.sh
./scripts/docker-run.sh -- ./scripts/capture-sass.sh
```

The input is `ptx/smoke/smoke_axpy.cu`. Outputs are:

```text
.bcodex/ptx/smoke_axpy.ptx
.bcodex/ptx/smoke_axpy.cubin
.bcodex/ptx/smoke_axpy.sass
```

Inspect the compiler boundary rather than assuming the PTX maps one-to-one to
machine instructions:

```bash
less .bcodex/ptx/smoke_axpy.ptx
less .bcodex/ptx/smoke_axpy.sass
```

The scripts exit with a diagnostic when `nvcc`, `ptxas`, `cuobjdump`, the
input artifact, the `sm_120` target, or the expected kernel entry is missing.

## PyPTX development workflow

PyPTX 0.1.1 is an optional development generator and transpiler. It is not a
runtime dependency: do not import it from `ptxsplat/`, select a backend based
on its availability, or require users to install it. A successful experiment
must end as reviewed PTX embedded or loaded by the native extension, with the
reference backend retained as the correctness oracle.

Install the exact authoring version into the dedicated persistent state mount,
then transpile the smoke PTX. The image intentionally does not include
`ensurepip`, so use `pip --target` instead of `venv`:

```bash
./scripts/docker-run.sh -- python3 -m pip install \
  --target /ptxsplat-state/pyptx-0.1.1 pyptx==0.1.1
```

The repository wrapper performs the version check and output validation when
the active `python3` is the PyPTX environment:

```bash
./scripts/docker-run.sh -- bash -lc \
  'PYTHONPATH=/ptxsplat-state/pyptx-0.1.1 \
   exec ./scripts/ptx-smoke.sh --with-pyptx'
```

The wrapper transpiles PTX to an editable Python generator, re-emits PTX from
that generator, assembles the re-emitted result, and requires its SASS to match
the unsugared NVCC baseline. Run the two authoring stages independently with:

```bash
./scripts/docker-run.sh -- bash -lc \
  'PYTHONPATH=/ptxsplat-state/pyptx-0.1.1 \
   ./scripts/ptx-transpile.sh && ./scripts/ptx-emit.sh'
```

Treat transpiled Python as an iteration artifact. Diff its re-emitted PTX,
assemble that PTX for `sm_120`, inspect SASS, run parity tests, then benchmark.
PyPTX must not remain on the launch path of an accepted runtime kernel.
The smoke path omits `--sugar` so symbol names and structure remain a stable
baseline. Apply PyPTX's sugar pass only in a separate experiment and verify the
re-emitted PTX before accepting any refactor.

## Optimization loop

1. Capture a reference benchmark and Nsight Compute profile using the fixed
   scenes and protocol in `docs/BENCHMARKING.md`.
2. Identify a measured bottleneck and change one kernel property at a time.
3. Assemble for `sm_120`, inspect register/spill output and SASS, then run the
   full correctness gate before timing.
4. Record accepted and rejected results under `benchmark-results/`.
5. Freeze only a reproducible winner; unsupported shapes continue to use the
   reference backend in `auto` mode.

Use `scripts/docker-run.sh --profile` only for Nsight Compute. PTX generation,
assembly, and SASS inspection do not require `CAP_SYS_ADMIN`.
