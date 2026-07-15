"""Small, inspectable scenes for localizing 3DGS rendering failures."""

from importlib import metadata
from math import cos, pi, sin

import pytest
import torch

from ptxsplat.cuda._wrapper import rasterize_to_pixels, spherical_harmonics
from ptxsplat.rendering import rasterization


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
DEVICE = torch.device("cuda:0")
WIDTH = HEIGHT = 32
FOCAL = 24.0


def _camera(cx=16.0, cy=16.0, focal=FOCAL):
    viewmats = torch.eye(4, device=DEVICE)[None]
    Ks = torch.tensor(
        [[[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]]],
        device=DEVICE,
    )
    return viewmats, Ks


def _scene(means, *, scales=None, quats=None, opacities=None, colors=None):
    means = torch.tensor(means, dtype=torch.float32, device=DEVICE).reshape(-1, 3)
    n = len(means)
    if scales is None:
        scales = [[0.18, 0.18, 0.18]] * n
    if quats is None:
        quats = [[1.0, 0.0, 0.0, 0.0]] * n
    if opacities is None:
        opacities = [0.6] * n
    if colors is None:
        basis = [[1.0, 0.1, 0.1], [0.1, 1.0, 0.1], [0.1, 0.1, 1.0]]
        colors = [basis[i % len(basis)] for i in range(n)]
    return {
        "means": means,
        "quats": torch.tensor(quats, dtype=torch.float32, device=DEVICE).reshape(n, 4),
        "scales": torch.tensor(scales, dtype=torch.float32, device=DEVICE).reshape(
            n, 3
        ),
        "opacities": torch.tensor(opacities, dtype=torch.float32, device=DEVICE),
        "colors": torch.tensor(colors, dtype=torch.float32, device=DEVICE).reshape(
            n, 3
        ),
    }


def _render(scene, *, background=None, camera=None, **kwargs):
    viewmats, Ks = _camera() if camera is None else camera
    backgrounds = None
    if background is not None:
        backgrounds = torch.tensor([background], dtype=torch.float32, device=DEVICE)
    return rasterization(
        **scene,
        viewmats=viewmats,
        Ks=Ks,
        width=WIDTH,
        height=HEIGHT,
        packed=True,
        backgrounds=backgrounds,
        **kwargs,
    )


def _assert_image_contract(render, alpha):
    assert render.shape == (1, HEIGHT, WIDTH, 3)
    assert alpha.shape == (1, HEIGHT, WIDTH, 1)
    assert torch.isfinite(render).all()
    assert torch.isfinite(alpha).all()
    assert ((0.0 <= alpha) & (alpha <= 1.0)).all()


def _moments(alpha):
    weights = alpha[0, ..., 0]
    ys, xs = torch.meshgrid(
        torch.arange(HEIGHT, device=DEVICE) + 0.5,
        torch.arange(WIDTH, device=DEVICE) + 0.5,
        indexing="ij",
    )
    total = weights.sum()
    assert total > 0
    mx = (weights * xs).sum() / total
    my = (weights * ys).sum() / total
    return (weights * (xs - mx).square()).sum() / total, (
        weights * (ys - my).square()
    ).sum() / total


def test_01_empty_and_fully_culled_are_background_only():
    background = [0.2, 0.3, 0.4]
    for scene in (_scene([]), _scene([[0.0, 0.0, -1.0]])):
        render, alpha, _ = _render(scene, background=background)
        expected = torch.tensor(background, device=DEVICE).expand_as(render)
        torch.testing.assert_close(render, expected)
        torch.testing.assert_close(alpha, torch.zeros_like(alpha))


def test_02_centered_isotropic_gaussian_is_symmetric():
    render, alpha, meta = _render(_scene([[0.0, 0.0, 2.0]]))
    _assert_image_contract(render, alpha)
    torch.testing.assert_close(
        meta["means2d"][0], torch.tensor([16.0, 16.0], device=DEVICE)
    )
    center = alpha[0, 15:17, 15:17, 0]
    torch.testing.assert_close(
        center, center[0, 0].expand_as(center), atol=1e-6, rtol=1e-6
    )


def test_03_off_center_projection_matches_pinhole_formula():
    scene = _scene([[0.25, -0.125, 2.0]])
    _, _, meta = _render(scene)
    expected = torch.tensor([19.0, 14.5], device=DEVICE)
    torch.testing.assert_close(meta["means2d"][0], expected, atol=1e-5, rtol=0)


def test_04_intrinsics_transform_projected_mean():
    scene = _scene([[0.25, -0.125, 2.0]])
    _, _, meta = _render(scene, camera=_camera(cx=9.0, cy=11.0, focal=40.0))
    expected = torch.tensor([14.0, 8.5], device=DEVICE)
    torch.testing.assert_close(meta["means2d"][0], expected, atol=1e-5, rtol=0)


def test_05_axis_aligned_anisotropy_has_expected_orientation():
    scene = _scene([[0.0, 0.0, 2.0]], scales=[[0.35, 0.08, 0.08]])
    _, alpha, _ = _render(scene)
    var_x, var_y = _moments(alpha)
    assert var_x > 4.0 * var_y


def test_06_quarter_turn_swaps_anisotropic_axes():
    q = [cos(pi / 4), 0.0, 0.0, sin(pi / 4)]
    scene = _scene([[0.0, 0.0, 2.0]], scales=[[0.35, 0.08, 0.08]], quats=[q])
    _, alpha, _ = _render(scene)
    var_x, var_y = _moments(alpha)
    assert var_y > 4.0 * var_x


def test_07_image_edge_clipping_is_bounded():
    render, alpha, _ = _render(_scene([[-1.30, 0.0, 2.0]], scales=[[0.4, 0.4, 0.4]]))
    _assert_image_contract(render, alpha)
    assert alpha.sum() > 0
    assert alpha[:, :, -1].max().item() == 0.0


def test_08_tile_boundary_has_no_omission_or_duplicate():
    _, alpha, _ = _render(_scene([[0.0, 0.0, 2.0]], scales=[[0.3, 0.3, 0.3]]))
    torch.testing.assert_close(alpha[0, :, 15], alpha[0, :, 16], atol=1e-6, rtol=1e-6)


def test_09_separated_splats_keep_local_color_dominance():
    scene = _scene(
        [[-0.5, 0.0, 2.0], [0.5, 0.0, 2.0]],
        scales=[[0.12, 0.12, 0.12]] * 2,
        colors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    render, _, _ = _render(scene)
    assert render[0, 15, 9, 0] > render[0, 15, 9, 1]
    assert render[0, 15, 21, 1] > render[0, 15, 21, 0]


def test_10_overlapping_splats_follow_exact_front_to_back_compositing():
    means2d = torch.tensor([[[0.5, 0.5], [0.5, 0.5]]], device=DEVICE)
    conics = torch.tensor([[[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]], device=DEVICE)
    colors = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]], device=DEVICE)
    opacities = torch.tensor([[0.25, 0.5]], device=DEVICE)
    background = torch.tensor([[0.2, 0.4, 0.6]], device=DEVICE)
    offsets = torch.zeros((1, 1, 1), dtype=torch.int32, device=DEVICE)
    ids = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)

    render, alpha = rasterize_to_pixels(
        means2d, conics, colors, opacities, 1, 1, 16, offsets, ids, background
    )
    transmittance = (1.0 - opacities[0, 0]) * (1.0 - opacities[0, 1])
    expected = (
        colors[0, 0] * opacities[0, 0]
        + colors[0, 1] * (1.0 - opacities[0, 0]) * opacities[0, 1]
        + background[0] * transmittance
    )
    torch.testing.assert_close(render[0, 0, 0], expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        alpha[0, 0, 0, 0], 1.0 - transmittance, atol=1e-6, rtol=1e-6
    )


def test_11_distinct_depth_input_permutation_is_invariant():
    scene = _scene([[0.0, 0.0, 1.5], [0.0, 0.0, 2.5]])
    render, alpha, _ = _render(scene)
    permutation = torch.tensor([1, 0], device=DEVICE)
    permuted = {key: value[permutation] for key, value in scene.items()}
    other_render, other_alpha, _ = _render(permuted)
    torch.testing.assert_close(render, other_render, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(alpha, other_alpha, atol=1e-6, rtol=1e-6)


def test_12_zero_and_near_zero_opacity_are_distinguishable():
    background = [0.2, 0.3, 0.4]
    zero, zero_alpha, _ = _render(
        _scene([[0.0, 0.0, 2.0]], opacities=[0.0]), background=background
    )
    near, near_alpha, _ = _render(
        _scene([[0.0, 0.0, 2.0]], opacities=[0.01]), background=background
    )
    torch.testing.assert_close(
        zero, torch.tensor(background, device=DEVICE).expand_as(zero)
    )
    assert zero_alpha.max().item() == 0.0
    assert near_alpha.max() > 0
    assert not torch.equal(near, zero)


def test_13_opaque_foreground_suppresses_rear_color_gradient():
    scene = _scene(
        [[0.0, 0.0, 1.5], [0.0, 0.0, 2.0]],
        opacities=[1.0, 0.8],
        colors=[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    )
    scene["colors"].requires_grad_()
    # A principal point at (15.5, 15.5) puts both means exactly on pixel (15, 15),
    # making the foreground alpha hit the renderer's 0.999 clamp.
    render, _, _ = _render(scene, camera=_camera(cx=15.5, cy=15.5))
    render[0, 15, 15].sum().backward()
    front = scene["colors"].grad[0].norm()
    rear = scene["colors"].grad[1].norm()
    assert rear < 0.01 * front


def test_14_near_and_far_plane_clipping_are_one_sided():
    scene = _scene(
        [
            [0.0, 0.0, 0.009],
            [0.0, 0.0, 0.01],
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 5.001],
        ]
    )
    _, _, meta = _render(scene, near_plane=0.01, far_plane=5.0)
    assert meta["gaussian_ids"].tolist() == [1, 2]


def test_15_background_uses_final_transmittance():
    scene = _scene([[0.0, 0.0, 2.0]])
    plain, alpha, _ = _render(scene)
    background = [0.2, 0.3, 0.4]
    composed, other_alpha, _ = _render(scene, background=background)
    expected = plain + (1.0 - alpha) * torch.tensor(background, device=DEVICE)
    torch.testing.assert_close(composed, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(alpha, other_alpha)


def test_16_quaternion_normalization_and_degree_zero_sh_invariants():
    scene = _scene(
        [[0.0, 0.0, 2.0]],
        scales=[[0.35, 0.08, 0.08]],
        quats=[[1.0, 0.2, 0.3, 0.4]],
    )
    scaled = {key: value.clone() for key, value in scene.items()}
    scaled["quats"] *= 7.0
    render, alpha, _ = _render(scene)
    other_render, other_alpha, _ = _render(scaled)
    torch.testing.assert_close(render, other_render, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(alpha, other_alpha, atol=1e-5, rtol=1e-5)

    dirs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], device=DEVICE)
    coeffs = torch.ones((2, 1, 3), device=DEVICE)
    values = spherical_harmonics(0, dirs, coeffs)
    torch.testing.assert_close(values, torch.full_like(values, 0.28209479177387814))


def test_exact_compositing_backward_weights():
    means2d = torch.tensor(
        [[[0.5, 0.5], [0.5, 0.5]]], device=DEVICE, requires_grad=True
    )
    conics = torch.tensor(
        [[[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]],
        device=DEVICE,
        requires_grad=True,
    )
    colors = torch.tensor(
        [[[1.0, 0.2, 0.0], [0.0, 0.1, 1.0]]],
        device=DEVICE,
        requires_grad=True,
    )
    opacities = torch.tensor([[0.25, 0.5]], device=DEVICE, requires_grad=True)
    background = torch.tensor([[0.2, 0.4, 0.6]], device=DEVICE, requires_grad=True)
    offsets = torch.zeros((1, 1, 1), dtype=torch.int32, device=DEVICE)
    ids = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    upstream = torch.tensor([0.3, -0.2, 0.7], device=DEVICE)
    alpha_upstream = 0.4

    render, alpha = rasterize_to_pixels(
        means2d, conics, colors, opacities, 1, 1, 16, offsets, ids, background
    )
    loss = render[0, 0, 0].dot(upstream) + alpha[0, 0, 0, 0] * alpha_upstream
    loss.backward()

    o0, o1 = opacities.detach()[0]
    c0, c1 = colors.detach()[0]
    bg = background.detach()[0]
    torch.testing.assert_close(colors.grad[0, 0], upstream * o0, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        colors.grad[0, 1], upstream * (1 - o0) * o1, atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        background.grad[0],
        upstream * (1 - o0) * (1 - o1),
        atol=1e-6,
        rtol=1e-6,
    )
    expected_o0 = upstream.dot(c0 - c1 * o1 - bg * (1 - o1)) + alpha_upstream * (1 - o1)
    expected_o1 = upstream.dot((1 - o0) * (c1 - bg)) + alpha_upstream * (1 - o0)
    torch.testing.assert_close(
        opacities.grad[0],
        torch.stack([expected_o0, expected_o1]),
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        means2d.grad, torch.zeros_like(means2d), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(conics.grad, torch.zeros_like(conics), atol=1e-6, rtol=0)


def test_upstream_gsplat_153_forward_parity_when_installed():
    try:
        version = metadata.version("gsplat")
    except metadata.PackageNotFoundError:
        pytest.skip("official gsplat is not installed")
    if version != "1.5.3":
        pytest.skip(f"requires gsplat 1.5.3, found {version}")

    from gsplat.rendering import rasterization as upstream_rasterization

    scene = _scene([[-0.2, 0.1, 1.5], [0.25, -0.15, 2.5]])
    viewmats, Ks = _camera()
    ours = _render(scene)[:2]
    upstream = upstream_rasterization(
        **scene, viewmats=viewmats, Ks=Ks, width=WIDTH, height=HEIGHT, packed=True
    )[:2]
    for actual, expected in zip(ours, upstream):
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
