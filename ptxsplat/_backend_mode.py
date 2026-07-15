"""Backend selection for reference and architecture-specialized kernels."""

from __future__ import annotations

import os
from enum import Enum

import torch


class Backend(str, Enum):
    AUTO = "auto"
    REFERENCE = "reference"
    SM120 = "sm120"


def requested_backend() -> Backend:
    value = os.getenv("PTXSPLAT_BACKEND", Backend.AUTO.value).lower()
    try:
        return Backend(value)
    except ValueError as exc:
        choices = ", ".join(backend.value for backend in Backend)
        raise ValueError(
            f"Invalid PTXSPLAT_BACKEND={value!r}; expected one of: {choices}"
        ) from exc


def resolve_backend(device: torch.device) -> Backend:
    """Resolve the implementation for a call without silently forcing SM120."""

    requested = requested_backend()
    if requested is Backend.REFERENCE:
        return requested

    if requested is Backend.SM120:
        if device.type != "cuda" or torch.cuda.get_device_capability(device) != (12, 0):
            raise RuntimeError("PTXSPLAT_BACKEND=sm120 requires an SM120 CUDA device")
        return requested

    if device.type == "cuda" and torch.cuda.get_device_capability(device) == (12, 0):
        return Backend.SM120
    return Backend.REFERENCE
