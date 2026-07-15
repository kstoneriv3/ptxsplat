# Correctness contract

Optimized kernels are compared with the unchanged gsplat 1.5.3-derived backend
and, where practical, an independent FP64 CPU formula or invariant. Diagnostic
images are at most 32 by 32 pixels and scenes contain at most eight Gaussians.

## Diagnostic scenes

1. Empty and fully culled scenes produce only background and zero alpha.
2. A centered isotropic Gaussian has radial symmetry and analytic weights.
3. An off-center Gaussian projects to the expected image coordinate.
4. Focal-length and principal-point changes transform the projected mean.
5. Axis-aligned anisotropy produces the expected ellipse orientation.
6. A 90-degree rotation swaps the anisotropic ellipse axes.
7. Image-edge clipping cannot write outside the image or alter interior values.
8. A splat crossing tile boundaries has no seams, omissions, or duplicates.
9. Separated splats do not interfere outside their support.
10. Two overlapping splats follow analytic front-to-back alpha compositing.
11. Permuting inputs with distinct depths preserves the rendered result.
12. Zero and near-zero opacity have the expected output and gradient behavior.
13. An opaque foreground suppresses rear contributions and gradients.
14. Near-plane, far-plane, and radius clipping use one-sided boundary cases.
15. Background compositing follows `color + transmittance * background`.
16. Quaternion normalization and selected SH basis directions match invariants.

## Comparison policy

- Forward tensors: `atol=1e-4`, `rtol=1e-4`.
- Gradient tensors: `atol=1e-4`, `rtol=1e-3`.
- Integer IDs, offsets, radii, and ordering: exact unless an explicitly
  documented equal-depth tie has no upstream ordering guarantee.
- Reject NaN, Inf, out-of-bounds access, unexplained nondeterminism, and any
  candidate requiring weaker tolerances.

Gradients cover means, quaternions, scales, opacities, RGB/SH coefficients,
backgrounds, and view matrices. Finite differences are evaluated only away
from clipping, sorting, and early-termination discontinuities.
