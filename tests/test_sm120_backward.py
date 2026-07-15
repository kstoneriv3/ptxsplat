import os

import pytest
import torch

from ptxsplat.cuda._wrapper import rasterize_to_pixels


DEVICE = torch.device("cuda:0")
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability(DEVICE) != (12, 0),
        reason="SM120 required",
    ),
]


def _inputs(packed: bool, empty: bool = False):
    images, count = 2, 6
    width, height, tile_size = 19, 17, 16
    base_means = torch.tensor(
        [
            [0.5, 0.5],
            [8.5, 7.5],
            [15.5, 15.5],
            [16.5, 0.5],
            [2.5, 16.5],
            [18.5, 16.5],
        ],
        device=DEVICE,
    )
    means = torch.stack(
        (base_means, base_means + torch.tensor([0.25, -0.25], device=DEVICE))
    )
    conics = torch.tensor(
        [
            [0.35, 0.01, 0.4],
            [0.08, 0.0, 0.1],
            [0.2, -0.01, 0.15],
            [0.5, 0.02, 0.45],
            [0.3, 0.0, 0.3],
            [0.6, -0.02, 0.55],
        ],
        device=DEVICE,
    ).expand(images, -1, -1).clone()
    colors = torch.linspace(0.05, 0.95, images * count * 3, device=DEVICE).reshape(
        images, count, 3
    )
    opacities = torch.tensor(
        [0.05, 0.2, 0.45, 0.7, 0.999, 0.3], device=DEVICE
    ).expand(images, -1).clone()
    flatten_ids = torch.empty(0, dtype=torch.int32, device=DEVICE)
    offsets = torch.zeros((images, 2, 2), dtype=torch.int32, device=DEVICE)
    if not empty:
        flatten_ids = torch.cat(
            [
                torch.arange(image * count, (image + 1) * count, device=DEVICE)
                for image in range(images)
                for _ in range(4)
            ]
        ).to(torch.int32)
        offsets.copy_(
            torch.arange(0, images * 4 * count, count, device=DEVICE).reshape(
                images, 2, 2
            )
        )
    backgrounds = torch.tensor(
        [[0.1, 0.2, 0.3], [0.6, 0.4, 0.2]], device=DEVICE
    )
    masks = torch.tensor(
        [[[False, True], [True, True]], [[False, True], [True, True]]],
        device=DEVICE,
    )
    if packed:
        means = means.reshape(-1, 2)
        conics = conics.reshape(-1, 3)
        colors = colors.reshape(-1, 3)
        opacities = opacities.reshape(-1)
    return {
        "means2d": means,
        "conics": conics,
        "colors": colors,
        "opacities": opacities,
        "image_width": width,
        "image_height": height,
        "tile_size": tile_size,
        "isect_offsets": offsets,
        "flatten_ids": flatten_ids,
        "backgrounds": backgrounds,
        "masks": masks,
        "packed": packed,
    }


def _run(backend: str, packed: bool, background: bool, absgrad: bool, empty: bool):
    previous = os.environ.get("PTXSPLAT_BACKEND")
    try:
        os.environ["PTXSPLAT_BACKEND"] = backend
        inputs = _inputs(packed, empty)
        if not background:
            inputs["backgrounds"] = None
        else:
            # The inherited masked-forward path does not initialize alpha for
            # disabled tiles, which makes its Python background gradient
            # intentionally unsuitable as a two-run oracle.
            inputs["masks"] = torch.ones_like(inputs["masks"])
        differentiable = ["means2d", "conics", "colors", "opacities"]
        for name in differentiable:
            inputs[name].requires_grad_(True)
        if inputs["backgrounds"] is not None:
            inputs["backgrounds"].requires_grad_(True)
            differentiable.append("backgrounds")
        render, alpha = rasterize_to_pixels(absgrad=absgrad, **inputs)
        pixel_mask = inputs["masks"].repeat_interleave(
            inputs["tile_size"], dim=-2
        ).repeat_interleave(inputs["tile_size"], dim=-1)
        pixel_mask = pixel_mask[..., : alpha.shape[-3], : alpha.shape[-2]]
        color_weight = torch.linspace(
            0.1, 0.9, render.numel(), device=DEVICE
        ).reshape_as(render)
        alpha_weight = torch.linspace(
            -0.2, 0.3, alpha.numel(), device=DEVICE
        ).reshape_as(alpha)
        loss = render.mul(color_weight).sum()
        loss = loss + alpha[..., 0][pixel_mask].mul(
            alpha_weight[..., 0][pixel_mask]
        ).sum()
        loss.backward()
        result = {
            "render": render.detach(),
            "alpha": torch.where(pixel_mask[..., None], alpha.detach(), 0.0),
            **{name: inputs[name].grad.detach() for name in differentiable},
        }
        if absgrad:
            result["absgrad"] = inputs["means2d"].absgrad.detach()
        return result
    finally:
        if previous is None:
            os.environ.pop("PTXSPLAT_BACKEND", None)
        else:
            os.environ["PTXSPLAT_BACKEND"] = previous


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("absgrad", [False, True])
def test_sm120_matches_reference_on_edge_matrix(packed, background, absgrad):
    reference = _run("reference", packed, background, absgrad, empty=False)
    candidate = _run("sm120", packed, background, absgrad, empty=False)
    assert candidate.keys() == reference.keys()
    for name in reference:
        torch.testing.assert_close(
            candidate[name], reference[name], rtol=2e-4, atol=2e-5
        )


@pytest.mark.parametrize("packed", [False, True])
def test_sm120_empty_intersections_match_reference(packed):
    reference = _run("reference", packed, True, True, empty=True)
    candidate = _run("sm120", packed, True, True, empty=True)
    for name in reference:
        torch.testing.assert_close(candidate[name], reference[name], rtol=0, atol=0)


def test_auto_matches_sm120_on_supported_call():
    automatic = _run("auto", True, True, True, empty=False)
    candidate = _run("sm120", True, True, True, empty=False)
    for name in candidate:
        torch.testing.assert_close(
            automatic[name], candidate[name], rtol=2e-4, atol=2e-5
        )


def test_sm120_rejects_unsupported_channel_count(monkeypatch):
    monkeypatch.setenv("PTXSPLAT_BACKEND", "sm120")
    inputs = _inputs(packed=True)
    inputs["colors"] = torch.cat(
        (inputs["colors"], torch.zeros_like(inputs["colors"][:, :1])), dim=-1
    ).requires_grad_(True)
    inputs["backgrounds"] = torch.cat(
        (inputs["backgrounds"], torch.zeros_like(inputs["backgrounds"][:, :1])),
        dim=-1,
    )
    render, _ = rasterize_to_pixels(**inputs)
    with pytest.raises(RuntimeError, match="supports only RGB"):
        render.sum().backward()


def test_auto_falls_back_for_unsupported_channel_count(monkeypatch):
    monkeypatch.setenv("PTXSPLAT_BACKEND", "auto")
    inputs = _inputs(packed=True)
    inputs["colors"] = torch.cat(
        (inputs["colors"], torch.zeros_like(inputs["colors"][:, :1])), dim=-1
    ).requires_grad_(True)
    inputs["backgrounds"] = torch.cat(
        (inputs["backgrounds"], torch.zeros_like(inputs["backgrounds"][:, :1])),
        dim=-1,
    )
    render, _ = rasterize_to_pixels(**inputs)
    render.sum().backward()
    assert inputs["colors"].grad is not None


def test_sm120_rejects_unsupported_tile_size(monkeypatch):
    monkeypatch.setenv("PTXSPLAT_BACKEND", "sm120")
    inputs = _inputs(packed=True)
    inputs["tile_size"] = 8
    inputs["isect_offsets"] = torch.zeros(
        (2, 3, 3), dtype=torch.int32, device=DEVICE
    )
    inputs["masks"] = torch.ones_like(inputs["isect_offsets"], dtype=torch.bool)
    inputs["flatten_ids"] = torch.empty(0, dtype=torch.int32, device=DEVICE)
    inputs["means2d"].requires_grad_(True)
    render, _ = rasterize_to_pixels(**inputs)
    with pytest.raises(RuntimeError, match="tile_size=16"):
        render.sum().backward()


def test_auto_falls_back_for_unsupported_tile_size(monkeypatch):
    monkeypatch.setenv("PTXSPLAT_BACKEND", "auto")
    inputs = _inputs(packed=True)
    inputs["tile_size"] = 8
    inputs["isect_offsets"] = torch.zeros(
        (2, 3, 3), dtype=torch.int32, device=DEVICE
    )
    inputs["masks"] = torch.ones_like(inputs["isect_offsets"], dtype=torch.bool)
    inputs["flatten_ids"] = torch.empty(0, dtype=torch.int32, device=DEVICE)
    inputs["means2d"].requires_grad_(True)
    render, _ = rasterize_to_pixels(**inputs)
    render.sum().backward()
    assert inputs["means2d"].grad is not None
