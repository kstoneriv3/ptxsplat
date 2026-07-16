import os

import pytest
import torch


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
    images, count = 2, 401
    width, height, tile_size = 19, 17, 16
    index = torch.arange(count, device=DEVICE, dtype=torch.float32)
    means = torch.stack(
        (
            index.remainder(width) + 0.5,
            (index * 7).remainder(height) + 0.5,
        ),
        dim=-1,
    )
    means = torch.stack((means, means + torch.tensor([0.125, -0.125], device=DEVICE)))
    conics = (
        torch.tensor([0.005, 0.0001, 0.006], device=DEVICE)
        .expand(images, count, 3)
        .clone()
    )
    colors = torch.linspace(0.01, 0.99, images * count * 3, device=DEVICE).reshape(
        images, count, 3
    )
    opacities = (
        torch.linspace(0.08, 0.22, count, device=DEVICE).expand(images, count).clone()
    )
    offsets = torch.zeros((images, 2, 2), dtype=torch.int32, device=DEVICE)
    flatten_ids = torch.empty(0, dtype=torch.int32, device=DEVICE)
    if not empty:
        flatten_ids = torch.cat(
            [
                torch.arange(
                    image * count,
                    (image + 1) * count,
                    device=DEVICE,
                    dtype=torch.int32,
                )
                for image in range(images)
                for _ in range(4)
            ]
        )
        offsets.copy_(
            torch.arange(
                0,
                images * 4 * count,
                count,
                device=DEVICE,
                dtype=torch.int32,
            ).reshape(images, 2, 2)
        )
    backgrounds = torch.tensor([[0.1, 0.2, 0.3], [0.7, 0.5, 0.25]], device=DEVICE)
    masks = torch.ones((images, 2, 2), dtype=torch.bool, device=DEVICE)
    if packed:
        means = means.reshape(-1, 2)
        conics = conics.reshape(-1, 3)
        colors = colors.reshape(-1, 3)
        opacities = opacities.reshape(-1)
    return {
        "means2d": means.contiguous(),
        "conics": conics.contiguous(),
        "colors": colors.contiguous(),
        "opacities": opacities.contiguous(),
        "backgrounds": backgrounds,
        "masks": masks,
        "image_width": width,
        "image_height": height,
        "tile_size": tile_size,
        "tile_offsets": offsets,
        "flatten_ids": flatten_ids,
    }


def _run(backend: str, inputs: dict):
    from ptxsplat.cuda._backend import _C

    previous_backend = os.environ.get("PTXSPLAT_BACKEND")
    previous_variant = os.environ.get("PTXSPLAT_SM120_FORWARD_VARIANT")
    try:
        os.environ["PTXSPLAT_BACKEND"] = backend
        os.environ["PTXSPLAT_SM120_FORWARD_VARIANT"] = (
            "reference" if backend == "reference" else "soa384"
        )
        outputs = _C.rasterize_to_pixels_3dgs_fwd(
            inputs["means2d"],
            inputs["conics"],
            inputs["colors"],
            inputs["opacities"],
            inputs["backgrounds"],
            inputs["masks"],
            inputs["image_width"],
            inputs["image_height"],
            inputs["tile_size"],
            inputs["tile_offsets"],
            inputs["flatten_ids"],
        )
        torch.cuda.synchronize()
        return tuple(output.clone() for output in outputs)
    finally:
        if previous_backend is None:
            os.environ.pop("PTXSPLAT_BACKEND", None)
        else:
            os.environ["PTXSPLAT_BACKEND"] = previous_backend
        if previous_variant is None:
            os.environ.pop("PTXSPLAT_SM120_FORWARD_VARIANT", None)
        else:
            os.environ["PTXSPLAT_SM120_FORWARD_VARIANT"] = previous_variant


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("background", [False, True])
def test_sm120_forward_and_last_ids_match_partial_batches(packed, background):
    inputs = _inputs(packed)
    if not background:
        inputs["backgrounds"] = None
    reference = _run("reference", inputs)
    candidate = _run("sm120", inputs)
    torch.testing.assert_close(candidate[0], reference[0], rtol=0, atol=0)
    torch.testing.assert_close(candidate[1], reference[1], rtol=0, atol=0)
    torch.testing.assert_close(candidate[2], reference[2], rtol=0, atol=0)
    assert (candidate[2] < inputs["flatten_ids"].numel()).all()


@pytest.mark.parametrize("packed", [False, True])
def test_sm120_forward_empty_intersections_match(packed):
    inputs = _inputs(packed, empty=True)
    reference = _run("reference", inputs)
    candidate = _run("sm120", inputs)
    for actual, expected in zip(candidate, reference):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("packed", [False, True])
def test_sm120_forward_auto_falls_back_for_non_rgb(packed):
    inputs = _inputs(packed)
    inputs["colors"] = inputs["colors"][..., :2].contiguous()
    inputs["backgrounds"] = inputs["backgrounds"][..., :2].contiguous()
    reference = _run("reference", inputs)
    candidate = _run("auto", inputs)
    for actual, expected in zip(candidate, reference):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_sm120_forward_masked_full_tile_matches_on_odd_image_edges():
    inputs = _inputs(packed=True)
    inputs["masks"][:, 0, 0] = False
    reference = _run("reference", inputs)
    candidate = _run("sm120", inputs)
    torch.testing.assert_close(candidate[0], reference[0], rtol=0, atol=0)

    enabled = inputs["masks"].repeat_interleave(16, -2).repeat_interleave(16, -1)
    enabled = enabled[:, : inputs["image_height"], : inputs["image_width"]]
    torch.testing.assert_close(
        candidate[1][enabled], reference[1][enabled], rtol=0, atol=0
    )
    torch.testing.assert_close(
        candidate[2][enabled], reference[2][enabled], rtol=0, atol=0
    )
