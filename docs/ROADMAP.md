# ptxsplat roadmap

## Objective

Build a faster, differentiable implementation of the gsplat 1.5.3 API. All
inherited features remain available through reference kernels; optimized paths
are added incrementally for SM120 and dispatched only after correctness and
performance gates pass.

## Phase 1: Foundation

- Rename the Python and native extension namespaces so `ptxsplat` and upstream
  `gsplat` can coexist.
- Preserve upstream history, attribution, tests, and reference CUDA kernels.
- Provide `auto`, `reference`, and `sm120` backend selection.
- Package a separate `ptxsplat-gsplat-overload` import compatibility layer.
- Establish the isolated RTX 5090 Docker environment.

## Phase 2: Correctness and Measurement

- Port the upstream suite and add the diagnostic scenes in `CORRECTNESS.md`.
- Measure projection, intersection generation, sorting, offset encoding, SH,
  raster forward, raster backward, and end-to-end training.
- Establish theoretical and sustained resource ceilings before selecting a hot
  kernel.

## Phase 3: Profile-Driven CUDA

- Optimize work partitioning, memory traffic, register pressure, occupancy, and
  compiler lowering for the highest-impact stage.
- Inspect emitted PTX and final SASS for every accepted candidate.
- Stop when three valid experiments improve the primary geometric mean by less
  than 1%, or the stage is within 10% of its sustained roof.

## Phase 4: Direct PTX

- Pin PyPTX 0.1.1 and use its PTX transpiler/DSL to turn the best CUDA kernel
  into an agent-editable SM120 instruction stream.
- Use PyPTX for rapid experiments, then freeze a winning PTX stream and launch
  it through the native extension to avoid Python custom-op overhead.
- Optimize raster forward, raster backward, and then projection/intersection in
  measured order.

## Phase 5: Coverage

Extend optimized dispatch to dense mode, camera batches, depth outputs,
antialiasing, absgrad, and sparse gradients. Keep reference fallback for camera
models, 2DGS, 3DGUT, and distributed paths until each receives its own parity
and performance matrix.

## Performance Gate

For reference latency `Tref` and measured sustained lower bound `Troof`, an
optimization milestone must close at least 25% of `Tref - Troof`; 50% is the
target. Remaining headroom and its limiting hardware resource must be reported.
