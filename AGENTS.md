# ptxsplat agent instructions

- Run CUDA builds, tests, and benchmarks through `scripts/docker-run.sh`; the
  target environment is `360-video-gs-dev:latest` on GPU 0 (RTX 5090, SM120).
- Use `scripts/docker-run.sh --profile` only for Nsight Compute work. It grants
  `CAP_SYS_ADMIN` inside a container that mounts only this repository and its
  dedicated state directory.
- Treat the reference backend and tests as the oracle. Do not weaken tolerances,
  remove adversarial cases, or change benchmark inputs to make a candidate pass.
- Every optimization must pass correctness before timing. Record accepted and
  rejected experiments as JSON under `benchmark-results/`; do not commit raw
  Nsight reports or generated binaries.
- Preserve the public gsplat 1.5.3-compatible API. Unsupported optimized shapes
  must fall back to the reference backend in `auto` mode.
- Do not claim a speedup without the protocol in `docs/BENCHMARKING.md`.
