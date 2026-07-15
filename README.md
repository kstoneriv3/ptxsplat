# ptxsplat

`ptxsplat` is an experimental, RTX-optimized fork of
[`gsplat`](https://github.com/nerfstudio-project/gsplat). It starts from gsplat
1.5.3 and keeps the reference CUDA implementation as a correctness oracle and
fallback while architecture-specific kernels are developed for NVIDIA SM120.

The first performance target is differentiable 3D Gaussian rendering on an RTX
5090: one pinhole camera, packed projection, classic RGB rasterization,
spherical harmonics through degree 3, backgrounds, and full backward gradients.

## Status

- Python API available as `ptxsplat`.
- Inherited gsplat 1.5.3 kernels are the current reference backend.
- `PTXSPLAT_BACKEND=auto` and `reference` select the reference backend.
- `PTXSPLAT_BACKEND=sm120` fails explicitly until an optimized kernel passes
  parity and benchmark gates.
- The optional `gsplat` import overload is packaged separately to keep upstream
  gsplat co-installable during development.

No performance claim is made until the benchmark protocol in
[`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) produces a reproducible result.

## Development Environment

The project uses the existing `360-video-gs-dev:latest` image, which contains
CUDA 12.8, PyTorch 2.9.1, Nsight Compute, and the RTX 5090 toolchain.

```bash
./scripts/docker-run.sh -- python3 -m pip install -e .
./scripts/docker-run.sh -- pytest -q tests
./scripts/run-codex-in-docker.sh
```

Nsight Compute needs GPU performance-counter permission on this host. Launch
the isolated container with the profiling capability only when required:

```bash
./scripts/run-codex-in-docker.sh --profile
./scripts/docker-run.sh --profile -- ncu --set full <command>
```

For unattended work, authenticate once and run the supervised launcher in a
detached host-side `tmux` session. It pins Sol/xhigh and retries failed CLI
runs without selecting a lower-effort fallback:

```bash
./scripts/docker-run.sh -- codex login --device-auth
tmux new-session -d -s ptxsplat \
  "cd $(pwd) && exec ./scripts/run-codex-supervised.sh"
```

The retry interval defaults to 30 minutes and can be changed with
`PTXSPLAT_CODEX_RETRY_SECONDS`. Logs are written to
`.bcodex/autonomous-codex.log`. The supervisor does not grant `SYS_ADMIN` by
default. Set `PTXSPLAT_CODEX_PROFILE=1` only for a run that needs Nsight Compute
performance counters.

The launcher mounts only this repository and a dedicated Codex/cache state
directory. It does not mount the host home, SSH configuration, datasets, or the
Docker socket.

## Installation

Install the development package:

```bash
python3 -m pip install -e .
```

The release extra installs a separately built compatibility distribution that
provides `gsplat.*` imports:

```bash
python3 -m pip install 'ptxsplat[gsplat_overload]'
```

Python package extras cannot satisfy another project's `Requires-Dist: gsplat`
metadata. Applications with a hard dependency on the official distribution
must adjust that dependency or install their application with dependency
resolution disabled after installing the overload.

## Documentation

- [`docs/ROADMAP.md`](docs/ROADMAP.md): implementation and optimization phases.
- [`docs/CORRECTNESS.md`](docs/CORRECTNESS.md): diagnostic scenes and parity
  policy.
- [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md): benchmark and roofline method.

## Attribution

The baseline implementation and much of the initial test suite come from
gsplat 1.5.3. Upstream copyright and Apache-2.0 license notices are retained.
See [`CITATION.bib`](CITATION.bib) for the upstream citation.
