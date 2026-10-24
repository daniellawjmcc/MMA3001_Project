
"""Model construction and checkpoint loading for MMA3001 experiments.

This module reconstructs YOLOv8n, RetinaNet and Faster R-CNN models
using existing experiment checkpoints.

No training, automatic pretrained-weight downloads or checkpoint
modifications are performed by these functions.
"""

from functools import partial
from pathlib import Path

import torch
import torch.nn as nn

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    retinanet_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

from .config import get_checkpoint_path


# Identify the detector architecture for each saved experiment.
MODEL_TYPES = {
    "yolo_baseline": "yolo",
    "yolo_extended": "yolo",
    "retinanet_baseline": "retinanet",
    "yolo_4class": "yolo",
    "retinanet_4class": "retinanet",
    "faster_rcnn_4class": "faster_rcnn",
}

# Record the number of foreground defect classes for each experiment.
NUM_DEFECT_CLASSES = {
    "yolo_baseline": 5,
    "yolo_extended": 5,
    "retinanet_baseline": 5,
    "yolo_4class": 4,
    "retinanet_4class": 4,
    "faster_rcnn_4class": 4,
}


def resolve_device(
    device: str | torch.device | None = None,
) -> torch.device:
    """Select a PyTorch execution device.

    Args:
        device: Explicit device or None for automatic CPU/GPU selection.

    Returns:
        Selected torch.device.

    Raises:
        ValueError: If CUDA is requested but unavailable.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    selected = torch.device(device)

    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA was requested but is not available."
        )

    return selected


def build_retinanet(num_defect_classes: int) -> nn.Module:
    """Build RetinaNet ResNet50-FPN V2 with the original classification head.

    This reproduces the GroupNorm classification-head configuration
    used in the preliminary RetinaNet experiment.

    Args:
        num_defect_classes: Number of foreground defect categories.

    Returns:
        RetinaNet model with one additional reserved class index.

    Raises:
        ValueError: If the number of classes is invalid.
    """
    if num_defect_classes <= 0:
        raise ValueError(
            "num_defect_classes must be positive."
        )

    # Disable pretrained-weight downloads because the checkpoint
    # will supply all model parameters.
    model = retinanet_resnet50_fpn_v2(
        weights=None,
        weights_backbone=None,
    )

    in_channels = model.backbone.out_channels

    num_anchors = (
        model.anchor_generator
        .num_anchors_per_location()[0]
    )

    # Match the original one-based defect-label convention.
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_defect_classes + 1,
        norm_layer=partial(nn.GroupNorm, 32),
    )

    return model


def build_faster_rcnn(num_defect_classes: int) -> nn.Module:
    """Build Faster R-CNN ResNet50-FPN V2 with a modified prediction head.

    Args:
        num_defect_classes: Number of foreground defect categories.

    Returns:
        Faster R-CNN model with a background class at index zero.

    Raises:
        ValueError: If the number of classes is invalid.
    """
    if num_defect_classes <= 0:
        raise ValueError(
            "num_defect_classes must be positive."
        )

    # Build the architecture without downloading pretrained weights.
    model = fasterrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=None,
    )

    # Replace the COCO prediction head with the project-specific head.
    in_features = (
        model.roi_heads.box_predictor.cls_score.in_features
    )

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_channels=in_features,
        num_classes=num_defect_classes + 1,
    )

    return model


def extract_state_dict(
    checkpoint: object,
) -> dict:
    """Extract model parameters from supported checkpoint formats.

    Supports checkpoints containing a direct state dictionary or a
    dictionary with model_state_dict or state_dict.

    Args:
        checkpoint: Deserialised trusted PyTorch checkpoint.

    Returns:
        Mapping containing model parameter tensors.

    Raises:
        ValueError: If the checkpoint has an unsupported structure.
    """
    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Expected a dictionary-based PyTorch checkpoint."
        )

    # Support common custom-training checkpoint wrappers.
    for key in ("model_state_dict", "state_dict"):
        if key in checkpoint:
            state_dict = checkpoint[key]

            if not isinstance(state_dict, dict):
                raise ValueError(
                    f"Checkpoint {key} is not a dictionary."
                )

            return state_dict

    # A directly saved state_dict maps parameter names to tensors.
    if checkpoint and all(
        isinstance(value, torch.Tensor)
        for value in checkpoint.values()
    ):
        return checkpoint

    raise ValueError(
        "Checkpoint does not contain a supported model state dictionary."
    )


def load_torchvision_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
) -> nn.Module:
    """Load existing weights into a Torchvision detector.

    Checkpoint loading uses strict parameter matching so architecture
    mismatches are reported rather than silently ignored.

    Args:
        model: Reconstructed Torchvision model architecture.
        checkpoint_path: Path to an existing PyTorch checkpoint.
        device: Inference device or None for automatic selection.

    Returns:
        Model with loaded parameters, moved to the device and in
        evaluation mode.

    Raises:
        FileNotFoundError: If the checkpoint is missing.
        ValueError: If the checkpoint format is unsupported.
        RuntimeError: If saved parameters do not match the model.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    selected_device = resolve_device(device)

    # Use weights_only for safer loading of trusted tensor checkpoints.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    state_dict = extract_state_dict(checkpoint)

    # Reject missing or unexpected parameter names.
    model.load_state_dict(state_dict, strict=True)

    model = model.to(selected_device)
    model.eval()

    return model


def load_yolo_checkpoint(
    checkpoint_path: str | Path,
):
    """Load an existing Ultralytics YOLO model.

    Args:
        checkpoint_path: Existing YOLO best.pt or last.pt checkpoint.

    Returns:
        Ultralytics YOLO object.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        ImportError: If Ultralytics is unavailable.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"YOLO checkpoint not found: {checkpoint_path}"
        )

    # Import Ultralytics only when a YOLO model is requested.
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise ImportError(
            "Ultralytics is required to load YOLO checkpoints. "
            "Install it with: pip install ultralytics"
        ) from error

    return YOLO(str(checkpoint_path))


def load_experiment_model(
    project_root: str | Path,
    run_name: str,
    device: str | torch.device | None = None,
):
    """Load a selected saved experiment model without retraining.

    Args:
        project_root: Project directory containing model_runs.
        run_name: Identifier defined in config.RUN_NAMES.
        device: Execution device for Torchvision models.

    Returns:
        Loaded YOLO or Torchvision detector ready for inference.

    Raises:
        ValueError: If the experiment identifier is unknown.
        FileNotFoundError: If the selected checkpoint is unavailable.
    """
    if run_name not in MODEL_TYPES:
        raise ValueError(
            f"Unknown model experiment: {run_name}"
        )

    checkpoint_path = get_checkpoint_path(
        project_root,
        run_name,
    )

    model_type = MODEL_TYPES[run_name]
    num_classes = NUM_DEFECT_CLASSES[run_name]

    # Ultralytics reconstructs its architecture from the checkpoint.
    if model_type == "yolo":
        return load_yolo_checkpoint(checkpoint_path)

    # Torchvision checkpoints require the matching architecture.
    if model_type == "retinanet":
        model = build_retinanet(num_classes)

    elif model_type == "faster_rcnn":
        model = build_faster_rcnn(num_classes)

    else:
        raise ValueError(
            f"Unsupported model architecture: {model_type}"
        )

    return load_torchvision_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )


def count_parameters(model) -> dict[str, int]:
    """Count total and trainable model parameters.

    Supports PyTorch models and Ultralytics YOLO objects.

    Args:
        model: PyTorch module or loaded Ultralytics YOLO object.

    Returns:
        Dictionary containing total and trainable parameter counts.

    Raises:
        TypeError: If the object has no supported PyTorch model.
    """
    # Ultralytics exposes the underlying PyTorch module as .model.
    if isinstance(model, nn.Module):
        module = model
    elif isinstance(getattr(model, "model", None), nn.Module):
        module = model.model
    else:
        raise TypeError(
            "Expected a PyTorch module or Ultralytics YOLO model."
        )

    total = sum(
        parameter.numel()
        for parameter in module.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )

    return {
        "Total Parameters": int(total),
        "Trainable Parameters": int(trainable),
    }
