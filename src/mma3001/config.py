
"""Project configuration and reproducibility utilities.

This module provides shared settings for the MMA3001 object detection
experiments. Datasets and trained models may be stored outside the
GitHub repository, including on Google Drive.
"""

from pathlib import Path
import os
import random

import numpy as np
import torch


SEED = 42

# Names used for the final four-class experiments.
FOUR_CLASS_NAMES = (
    "loose-meat",
    "packaging-error",
    "twisted-meat",
    "unsealed",
)

# Original preliminary experiments also included wrinkle.
FIVE_CLASS_NAMES = FOUR_CLASS_NAMES + ("wrinkle",)

# Experiment folder names are fixed to match the original saved runs.
RUN_NAMES = {
    "yolo_baseline": "yolov8n_v1_baseline",
    "yolo_extended": "yolov8n_v2_60epochs",
    "retinanet_baseline": "retinanet_resnet50_v1",
    "yolo_4class": "yolov8n_4class_30epochs",
    "retinanet_4class": "retinanet_resnet50_4class",
    "faster_rcnn_4class": "faster_rcnn_resnet50_4class",
}


def set_seed(seed: int = SEED) -> None:
    """Set random seeds used in the experiments.

    Args:
        seed: Seed applied to Python, NumPy and PyTorch.

    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_project_root(custom_root: str | Path | None = None) -> Path:
    """Locate the directory containing experiment datasets and runs.

    Args:
        custom_root: Optional explicit project location. If omitted,
            the MMA3001_PROJECT_ROOT environment variable is checked.

    Returns:
        Resolved directory containing the project data.

    Raises:
        FileNotFoundError: If no valid project directory is found.
    """
    explicit_root = custom_root or os.environ.get(
        "MMA3001_PROJECT_ROOT"
    )

    if explicit_root:
        root = Path(explicit_root).expanduser().resolve()

        if (root / "dataset").is_dir() or (root / "model_runs").is_dir():
            return root

        raise FileNotFoundError(
            f"Configured project directory is invalid: {root}"
        )

    candidates = [
        Path("/content/drive/MyDrive/MMA3001_Project"),
        Path("/content/gdrive_new/MyDrive/MMA3001_Project"),
        Path("/content/gdrive_mma3001/MyDrive/MMA3001_Project"),
        Path.cwd() / "MMA3001_Project",
        Path.cwd(),
        Path.cwd().parent,
    ]

    for candidate in candidates:
        if (
            (candidate / "dataset").is_dir()
            or (candidate / "model_runs").is_dir()
        ):
            return candidate.resolve()

    raise FileNotFoundError(
        "Experiment data directory not found. Set "
        "MMA3001_PROJECT_ROOT or mount the dataset storage."
    )


def get_dataset_path(
    project_root: str | Path,
    four_class: bool = True,
) -> Path:
    """Return the directory of an existing dataset.

    Args:
        project_root: Root directory containing the dataset folder.
        four_class: Whether to use the final four-class dataset.

    Returns:
        Directory containing the selected dataset.

    Raises:
        FileNotFoundError: If the dataset is missing.
    """
    name = (
        "pork_rasher_v4_4class"
        if four_class
        else "pork_rasher_v4_clean"
    )

    dataset_path = Path(project_root) / "dataset" / name

    if not (dataset_path / "data.yaml").is_file():
        raise FileNotFoundError(
            f"Dataset configuration not found: {dataset_path}"
        )

    return dataset_path.resolve()


def get_run_path(project_root: str | Path, run_name: str) -> Path:
    """Return a previously saved experiment directory.

    Args:
        project_root: Root directory containing the model_runs folder.
        run_name: Experiment identifier from RUN_NAMES.

    Returns:
        Existing experiment directory.

    Raises:
        ValueError: If the experiment identifier is unknown.
        FileNotFoundError: If the saved run directory does not exist.
    """
    if run_name not in RUN_NAMES:
        raise ValueError(
            f"Unknown experiment: {run_name}. "
            f"Available experiments: {list(RUN_NAMES)}"
        )

    run_path = (
        Path(project_root)
        / "model_runs"
        / RUN_NAMES[run_name]
    )

    if not run_path.is_dir():
        raise FileNotFoundError(
            f"Saved experiment not found: {run_path}"
        )

    return run_path.resolve()


def get_checkpoint_path(
    project_root: str | Path,
    run_name: str,
) -> Path:
    """Locate the selected checkpoint of a saved experiment.

    Args:
        project_root: Root directory containing experiment runs.
        run_name: Experiment identifier from RUN_NAMES.

    Returns:
        Path to the existing best.pt checkpoint.

    Raises:
        FileNotFoundError: If the checkpoint cannot be located.
    """
    run_path = get_run_path(project_root, run_name)

    # Ultralytics saves checkpoints in a weights subdirectory.
    if run_name.startswith("yolo"):
        checkpoint = run_path / "weights" / "best.pt"
    else:
        checkpoint = run_path / "best.pt"

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    return checkpoint.resolve()
