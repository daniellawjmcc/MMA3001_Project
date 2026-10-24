
"""Unit tests for model loading and checkpoint utilities."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

import mma3001.models as models

from mma3001.models import (
    resolve_device,
    extract_state_dict,
    load_torchvision_checkpoint,
    load_yolo_checkpoint,
    load_experiment_model,
    count_parameters,
)


def test_resolve_cpu_device():
    """An explicitly requested CPU device should be supported."""
    device = resolve_device("cpu")

    assert device.type == "cpu"


def test_invalid_cuda_device(monkeypatch):
    """Reject explicit CUDA selection when no GPU is available."""
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: False,
    )

    with pytest.raises(ValueError):
        resolve_device("cuda")


def test_extract_direct_state_dict():
    """Read directly saved PyTorch parameter dictionaries."""
    model = nn.Linear(4, 2)
    state_dict = model.state_dict()

    result = extract_state_dict(state_dict)

    assert set(result) == set(state_dict)


def test_extract_wrapped_state_dict():
    """Read parameters inside a checkpoint metadata dictionary."""
    model = nn.Linear(4, 2)

    checkpoint = {
        "epoch": 5,
        "model_state_dict": model.state_dict(),
    }

    result = extract_state_dict(checkpoint)

    assert "weight" in result
    assert "bias" in result


def test_invalid_checkpoint_structure():
    """Reject unsupported checkpoint structures."""
    with pytest.raises(ValueError):
        extract_state_dict({
            "epoch": 5,
            "accuracy": 0.5,
        })


def test_load_torchvision_checkpoint(tmp_path: Path):
    """Verify exact weight restoration and evaluation mode."""
    original_model = nn.Linear(4, 2)
    checkpoint_path = tmp_path / "best.pt"

    # Save a small trusted state dictionary.
    torch.save(
        original_model.state_dict(),
        checkpoint_path,
    )

    restored_model = nn.Linear(4, 2)

    loaded = load_torchvision_checkpoint(
        restored_model,
        checkpoint_path,
        device="cpu",
    )

    assert not loaded.training

    # Verify that every loaded tensor matches its saved value.
    for name, parameter in original_model.state_dict().items():
        assert torch.equal(
            parameter,
            loaded.state_dict()[name],
        )


def test_missing_torchvision_checkpoint(tmp_path: Path):
    """Missing checkpoints should produce a clear error."""
    model = nn.Linear(4, 2)

    with pytest.raises(FileNotFoundError):
        load_torchvision_checkpoint(
            model,
            tmp_path / "missing.pt",
            device="cpu",
        )


def test_checkpoint_architecture_mismatch(tmp_path: Path):
    """Reject incompatible checkpoint parameter dimensions."""
    original_model = nn.Linear(4, 2)
    incompatible_model = nn.Linear(4, 3)

    checkpoint_path = tmp_path / "best.pt"

    torch.save(
        original_model.state_dict(),
        checkpoint_path,
    )

    with pytest.raises(RuntimeError):
        load_torchvision_checkpoint(
            incompatible_model,
            checkpoint_path,
            device="cpu",
        )


def test_missing_yolo_checkpoint(tmp_path: Path):
    """Check the path before requiring Ultralytics."""
    with pytest.raises(FileNotFoundError):
        load_yolo_checkpoint(
            tmp_path / "missing.pt"
        )


def test_count_parameters():
    """Verify total and trainable parameter counting."""
    model = nn.Linear(4, 2)

    result = count_parameters(model)

    # Four-by-two weights and two bias values.
    assert result["Total Parameters"] == 10
    assert result["Trainable Parameters"] == 10


def test_frozen_parameter_count():
    """Frozen parameters should remain part of the total count."""
    model = nn.Linear(4, 2)
    model.bias.requires_grad = False

    result = count_parameters(model)

    assert result["Total Parameters"] == 10
    assert result["Trainable Parameters"] == 8


def test_invalid_model_object():
    """Unsupported model objects should be rejected."""
    with pytest.raises(TypeError):
        count_parameters(object())


def test_unknown_experiment(tmp_path: Path):
    """Unknown experiment identifiers should be rejected."""
    with pytest.raises(ValueError):
        load_experiment_model(
            tmp_path,
            "unknown_experiment",
        )


def test_experiment_dispatch(monkeypatch, tmp_path: Path):
    """Verify that experiment loading selects the correct model family."""
    fake_checkpoint = tmp_path / "best.pt"
    fake_checkpoint.touch()

    fake_model = nn.Linear(4, 2)

    # Avoid constructing a large detector during this unit test.
    monkeypatch.setattr(
        models,
        "get_checkpoint_path",
        lambda project_root, run_name: fake_checkpoint,
    )

    monkeypatch.setattr(
        models,
        "build_retinanet",
        lambda num_classes: fake_model,
    )

    loader = MagicMock(return_value=fake_model)

    monkeypatch.setattr(
        models,
        "load_torchvision_checkpoint",
        loader,
    )

    result = load_experiment_model(
        project_root=tmp_path,
        run_name="retinanet_4class",
        device="cpu",
    )

    assert result is fake_model
    loader.assert_called_once()
