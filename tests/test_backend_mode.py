import pytest
import torch

from ptxsplat._backend_mode import Backend, requested_backend, resolve_backend


def test_backend_defaults_to_reference(monkeypatch):
    monkeypatch.delenv("PTXSPLAT_BACKEND", raising=False)
    assert requested_backend() is Backend.AUTO
    assert resolve_backend(torch.device("cpu")) is Backend.REFERENCE


def test_reference_backend_is_explicit(monkeypatch):
    monkeypatch.setenv("PTXSPLAT_BACKEND", "reference")
    assert resolve_backend(torch.device("cpu")) is Backend.REFERENCE


def test_invalid_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("PTXSPLAT_BACKEND", "typo")
    with pytest.raises(ValueError, match="Invalid PTXSPLAT_BACKEND"):
        requested_backend()


def test_sm120_backend_cannot_silently_fall_back(monkeypatch):
    monkeypatch.setenv("PTXSPLAT_BACKEND", "sm120")
    with pytest.raises(RuntimeError, match="requires an SM120 CUDA device"):
        resolve_backend(torch.device("cpu"))


def test_sm120_backend_resolves_on_target_device(monkeypatch):
    monkeypatch.setenv("PTXSPLAT_BACKEND", "sm120")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    assert resolve_backend(torch.device("cuda")) is Backend.SM120


def test_auto_backend_resolves_on_target_device(monkeypatch):
    monkeypatch.delenv("PTXSPLAT_BACKEND", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    assert resolve_backend(torch.device("cuda")) is Backend.SM120
