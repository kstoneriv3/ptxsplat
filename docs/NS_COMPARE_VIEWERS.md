# Nerfstudio Comparison Viewers

Use the focused launcher for the two local Nerfstudio comparison viewers:

```bash
scripts/ns-compare-viewers.sh up
scripts/ns-compare-viewers.sh status
scripts/ns-compare-viewers.sh logs
scripts/ns-compare-viewers.sh down
```

`up` recreates the stable containers `ptxsplat-ns-compare-upstream` and
`ptxsplat-ns-compare-sm120` with host networking on ports 7007 and 7008. The
viewer command sets `TORCHDYNAMO_DISABLE=1` only around `ns-viewer`, avoiding
Nerfstudio viewer-time `torch.compile` work without changing training,
benchmarks, or the Docker image.

The import isolation is intentional:

- upstream: `PYTHONPATH=/workspace/.bcodex/gsplat-1.5.3`,
  `results/ns-compare/upstream/tiny-synthetic/splatfacto/matched-1000/config.yml`,
  `http://localhost:7007`
- ptxsplat: `PYTHONPATH=/workspace/compat/gsplat_overload:/workspace`,
  `PTXSPLAT_BACKEND=sm120`,
  `results/ns-compare/ptxsplat/tiny-synthetic/splatfacto/matched-1000/config.yml`,
  `http://localhost:7008`
