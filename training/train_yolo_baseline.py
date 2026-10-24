
# Reproduce the original five-class YOLOv8n baseline experiments.

import argparse
import json
import sys

from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


# Import the shared MMA3001 configuration.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mma3001.config import (
    FIVE_CLASS_NAMES,
    SEED,
    get_dataset_path,
    set_seed,
)


# Original YOLOv8n experiment settings.
IMAGE_SIZE = 640
BATCH_SIZE = 16
WORKERS = 2
INITIAL_WEIGHTS = "yolov8n.pt"

EXPERIMENTS = {
    "baseline": {
        "epochs": 30,
        "original_run": "yolov8n_v1_baseline",
        "reproduction_run": "reproduction_yolov8n_v1_baseline",
    },
    "extended": {
        "epochs": 60,
        "original_run": "yolov8n_v2_60epochs",
        "reproduction_run": "reproduction_yolov8n_v2_60epochs",
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command-line settings for the YOLOv8n experiments.

    Returns:
        Namespace containing the selected experiment, project root
        and optional reproduction output directory.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the original five-class YOLOv8n "
            "baseline and extended-training experiments."
        )
    )

    parser.add_argument(
        "--experiment",
        type=str,
        choices=["baseline", "extended"],
        required=True,
        help="Run the original 30-epoch or 60-epoch experiment.",
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="Project directory containing the original dataset.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional new directory for reproduction outputs.",
    )

    return parser.parse_args()


def verify_dataset(dataset_root: Path) -> Path:
    """Verify the five-class dataset and its YOLO configuration.

    Args:
        dataset_root: Directory containing the original cleaned
            five-class object detection dataset.

    Returns:
        Path to the verified `data.yaml` configuration file.

    Raises:
        FileNotFoundError: If the dataset or its YAML configuration
            cannot be found.
        ValueError: If the configured classes do not match the
            original five-class experiment.
    """
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Original dataset directory not found: {dataset_root}"
        )

    yaml_path = dataset_root / "data.yaml"

    if not yaml_path.is_file():
        raise FileNotFoundError(
            f"Dataset configuration missing: {yaml_path}"
        )

    with yaml_path.open("r", encoding="utf-8") as file:
        data_config = yaml.safe_load(file)

    configured_names = data_config.get("names")

    if isinstance(configured_names, dict):
        configured_names = [
            configured_names.get(
                index,
                configured_names.get(str(index))
            )
            for index in range(len(FIVE_CLASS_NAMES))
        ]

    if list(configured_names or []) != list(FIVE_CLASS_NAMES):
        raise ValueError(
            "Dataset class mapping does not match the original "
            f"five-class experiment: {configured_names}"
        )

    configured_count = data_config.get("nc")

    if (
        configured_count is not None
        and int(configured_count) != len(FIVE_CLASS_NAMES)
    ):
        raise ValueError(
            "Dataset class count differs from the original "
            "five-class experiment."
        )

    print("DATASET VERIFICATION")
    print("Dataset:", dataset_root)
    print("Configuration:", yaml_path)
    print("Classes:", configured_names)

    return yaml_path


def prepare_output_directory(
    project_root: Path,
    experiment: str,
    custom_output_dir: Path | None = None,
) -> Path:
    """Select a new training directory without overwriting saved runs.

    Args:
        project_root: Directory containing the project's datasets
            and original model runs.
        experiment: Selected experiment, either baseline or extended.
        custom_output_dir: Optional explicit reproduction directory.
            When omitted, a new directory under `model_runs` is used.

    Returns:
        Resolved path to the reproduction output directory.

    Raises:
        ValueError: If the output directory would overwrite or be
            placed inside a protected original experiment directory.
        FileExistsError: If the reproduction directory already exists.
    """
    config = EXPERIMENTS[experiment]

    if custom_output_dir is None:
        output_dir = (
            project_root
            / "model_runs"
            / config["reproduction_run"]
        )
    else:
        output_dir = custom_output_dir

    output_dir = output_dir.expanduser().resolve()

    protected_paths = [
        project_root.resolve(),
        (project_root / "dataset").resolve(),
        (project_root / "model_runs").resolve(),
    ]

    for experiment_config in EXPERIMENTS.values():
        protected_paths.append(
            (
                project_root
                / "model_runs"
                / experiment_config["original_run"]
            ).resolve()
        )


    for protected_path in protected_paths:
        protected_path = protected_path.resolve()

        # Allow new reproduction runs inside model_runs,
        # while protecting the original datasets and experiments.
        allowed_parent = (
            protected_path == project_root.resolve()
            or protected_path == (project_root / "model_runs").resolve()
        )

        if output_dir == protected_path or (
            not allowed_parent and protected_path in output_dir.parents
        ):
            raise ValueError(
                f"Output directory overlaps a protected location: "
                f"{protected_path}"
            )


    if output_dir.exists():
        raise FileExistsError(
            "The selected reproduction directory already exists: "
            f"{output_dir}"
        )

    # Ultralytics creates this directory when training starts.
    output_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir


def train_yolo(
    yaml_path: Path,
    output_dir: Path,
    experiment: str,
    device: int,
) -> Path:
    """Train YOLOv8n using the original experiment configuration.

    Each experiment starts from the pretrained YOLOv8n checkpoint,
    rather than continuing from the earlier experiment.

    Args:
        yaml_path: Path to the original five-class dataset YAML.
        output_dir: New directory for this reproduction run.
        experiment: Either baseline for 30 epochs or extended
            for 60 epochs.
        device: CUDA device index used for training.

    Returns:
        Path to the best model checkpoint saved during training.

    Raises:
        RuntimeError: If training completes without creating
            the expected best-model checkpoint.
    """
    config = EXPERIMENTS[experiment]

    model = YOLO(INITIAL_WEIGHTS)

    print("\nYOLOV8N TRAINING")
    print("Experiment:", experiment)
    print("Initial weights:", INITIAL_WEIGHTS)
    print("Epochs:", config["epochs"])
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Seed:", SEED)
    print("Workers:", WORKERS)
    print("Device:", device)
    print("Output directory:", output_dir)

    # Match the original Ultralytics training call.
    model.train(
        data=str(yaml_path),
        epochs=config["epochs"],
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=device,
        seed=SEED,
        workers=WORKERS,
        project=str(output_dir.parent),
        name=output_dir.name,
        plots=True,
        exist_ok=False,
    )

    best_checkpoint = (
        output_dir / "weights" / "best.pt"
    )

    if not best_checkpoint.is_file():
        raise RuntimeError(
            "Training finished but the best checkpoint "
            f"was not found: {best_checkpoint}"
        )

    print("\nTRAINING COMPLETED")
    print("Best checkpoint:", best_checkpoint)

    return best_checkpoint


def evaluate_yolo(
    checkpoint_path: Path,
    yaml_path: Path,
    output_dir: Path,
    device: int,
) -> dict:
    """Evaluate the best checkpoint on the validation split.

    The original five-class experiment evaluated the best
    checkpoint after training. This function saves the reported
    overall validation metrics for future reproducibility checks.

    Args:
        checkpoint_path: Path to the best saved YOLOv8n weights.
        yaml_path: Original five-class dataset YAML configuration.
        output_dir: Directory used to save validation results.
        device: CUDA device index used for inference.

    Returns:
        Dictionary containing precision, recall, mAP@50,
        mAP@50:95 and per-class mAP@50:95.
    """
    model = YOLO(str(checkpoint_path))

    # Run validation without creating an additional training run.
    metrics = model.val(
        data=str(yaml_path),
        split="val",
        imgsz=IMAGE_SIZE,
        device=device,
        workers=0,
        plots=True,
        project=str(output_dir),
        name="checkpoint_validation",
        exist_ok=False,
    )

    class_ap = {
        str(model.names[class_id]): float(ap)
        for class_id, ap in enumerate(metrics.box.maps)
    }

    results = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "per_class_mAP50_95": class_ap,
    }

    print("\nVALIDATION PERFORMANCE")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"mAP@50: {results['mAP50']:.4f}")
    print(f"mAP@50:95: {results['mAP50_95']:.4f}")

    print("\nPER-CLASS AP@50:95")

    for class_name, ap in class_ap.items():
        print(f"{class_name}: {ap:.4f}")

    with (output_dir / "validation_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    return results


def main() -> None:
    """Execute the selected YOLOv8n reproduction experiment.

    The original cleaned five-class dataset is used for training.
    Results are saved to a separate directory, protecting the
    previously trained baseline and extended YOLO checkpoints.

    Args:
        None. Settings are supplied through command-line arguments.

    Returns:
        None. Saves YOLO training artifacts, the best checkpoint
        and a JSON summary of validation metrics.

    Raises:
        RuntimeError: If CUDA is unavailable or the selected
            CUDA device cannot be used for training.
        FileNotFoundError: If the dataset or its YAML file is missing.
        ValueError: If the original class mapping is incorrect or
            an output path overlaps a protected location.
        FileExistsError: If the selected output directory exists.
    """
    args = parse_args()

    set_seed(SEED)

    project_root = (
        args.project_root.expanduser().resolve()
    )

    config = EXPERIMENTS[args.experiment]

    dataset_root = get_dataset_path(
        project_root,
        four_class=False,
    )

    yaml_path = verify_dataset(dataset_root)

    output_dir = prepare_output_directory(
        project_root=project_root,
        experiment=args.experiment,
        custom_output_dir=args.output_dir,
    )

    # The original training experiments used CUDA device zero.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required to reproduce the original "
            "YOLOv8n GPU training configuration."
        )

    device = 0

    if torch.cuda.device_count() <= device:
        raise RuntimeError(
            f"Requested CUDA device {device} is unavailable."
        )

    print("\nORIGINAL YOLO BASELINE CONFIGURATION")
    print("Experiment:", args.experiment)
    print("Original run:", config["original_run"])
    print("Dataset:", dataset_root)
    print("Output:", output_dir)

    best_checkpoint = train_yolo(
        yaml_path=yaml_path,
        output_dir=output_dir,
        experiment=args.experiment,
        device=device,
    )

    evaluate_yolo(
        checkpoint_path=best_checkpoint,
        yaml_path=yaml_path,
        output_dir=output_dir,
        device=device,
    )

    print("\nREPRODUCTION COMPLETED")
    print("Run directory:", output_dir)
    print("Best checkpoint:", best_checkpoint)
    print(
        "Validation results:",
        output_dir / "validation_metrics.json",
    )


if __name__ == "__main__":
    main()
