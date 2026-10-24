
# Reproduce the original four-class YOLOv8n training experiment.

import argparse
import json
import sys
import tempfile

from pathlib import Path

import torch
import yaml

from ultralytics import YOLO


# Import the shared MMA3001 project configuration.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mma3001.config import (
    FOUR_CLASS_NAMES,
    SEED,
    get_dataset_path,
    set_seed,
)


# Original experiment configuration.
EPOCHS = 30
IMAGE_SIZE = 640
BATCH_SIZE = 16
WORKERS = 0
INITIAL_WEIGHTS = "yolov8n.pt"

ORIGINAL_RUN = "yolov8n_4class_30epochs"
REPRODUCTION_RUN = "reproduction_yolov8n_4class_30epochs"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the YOLOv8n training experiment.

    Returns:
        Namespace containing the project root and optional custom
        reproduction output directory.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the original four-class YOLOv8n "
            "object detection training experiment."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="Project directory containing the four-class dataset.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory for reproduction results.",
    )

    return parser.parse_args()


def verify_dataset(dataset_root: Path) -> dict:
    """Verify the four-class dataset and original YOLO class mapping.

    The existing dataset configuration is inspected without
    modifying the original YAML or any annotation files.

    Args:
        dataset_root: Directory containing the existing four-class
            dataset and its `data.yaml` configuration.

    Returns:
        Dictionary containing the verified dataset configuration,
        including the four defect categories.

    Raises:
        FileNotFoundError: If the dataset, configuration file or
            required training and validation folders are missing.
        ValueError: If the configured class names or class count
            differ from the original four-class experiment.
    """
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Four-class dataset not found: {dataset_root}"
        )

    yaml_path = dataset_root / "data.yaml"

    if not yaml_path.is_file():
        raise FileNotFoundError(
            f"Dataset configuration not found: {yaml_path}"
        )

    required_directories = [
        dataset_root / "train" / "images",
        dataset_root / "train" / "labels",
        dataset_root / "valid" / "images",
        dataset_root / "valid" / "labels",
        dataset_root / "test" / "images",
        dataset_root / "test" / "labels",
    ]

    missing = [
        str(path)
        for path in required_directories
        if not path.is_dir()
    ]

    if missing:
        raise FileNotFoundError(
            "Required dataset directories are missing:\n"
            + "\n".join(missing)
        )

    with yaml_path.open("r", encoding="utf-8") as file:
        dataset_config = yaml.safe_load(file)

    if not isinstance(dataset_config, dict):
        raise ValueError(
            "The dataset YAML must contain a configuration mapping."
        )

    configured_names = dataset_config.get("names")

    if isinstance(configured_names, dict):
        configured_names = [
            configured_names.get(
                index,
                configured_names.get(str(index)),
            )
            for index in range(len(FOUR_CLASS_NAMES))
        ]

    expected_names = list(FOUR_CLASS_NAMES)

    if list(configured_names or []) != expected_names:
        raise ValueError(
            "Dataset classes differ from the original experiment.\n"
            f"Expected: {expected_names}\n"
            f"Found: {configured_names}"
        )

    configured_count = dataset_config.get("nc")

    if (
        configured_count is not None
        and int(configured_count) != len(expected_names)
    ):
        raise ValueError(
            "Dataset class count differs from the original "
            "four-class experiment."
        )

    print("DATASET VERIFICATION")
    print("Dataset:", dataset_root)
    print("Classes:", expected_names)
    print("Class count:", len(expected_names))

    return dataset_config


def prepare_output_directory(
    project_root: Path,
    custom_output_dir: Path | None = None,
) -> Path:
    """Select an unused output directory for reproduction training.

    The original model run and dataset directories are protected
    against accidental modification or checkpoint replacement.

    Args:
        project_root: Directory containing the project datasets
            and original model runs.
        custom_output_dir: Optional alternative output directory.
            If omitted, a reproduction folder under `model_runs`
            is selected.

    Returns:
        Resolved path to an unused reproduction output directory.

    Raises:
        ValueError: If the chosen output directory overlaps
            a protected dataset or original experiment directory.
        FileExistsError: If the reproduction output directory
            already exists.
    """
    if custom_output_dir is None:
        output_dir = (
            project_root
            / "model_runs"
            / REPRODUCTION_RUN
        )
    else:
        output_dir = custom_output_dir

    output_dir = output_dir.expanduser().resolve()

    protected_paths = [
        project_root.resolve(),
        (project_root / "dataset").resolve(),
        (project_root / "model_runs").resolve(),
        (
            project_root / "model_runs" / ORIGINAL_RUN
        ).resolve(),
    ]

    for protected_path in protected_paths:
        if (
            output_dir == protected_path
            or protected_path in output_dir.parents
        ):
            raise ValueError(
                "Output directory overlaps a protected project "
                f"location: {protected_path}"
            )

    if output_dir.exists():
        raise FileExistsError(
            f"Reproduction directory already exists: {output_dir}"
        )

    output_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir


def create_runtime_yaml(
    dataset_root: Path,
    dataset_config: dict,
    runtime_directory: Path,
) -> Path:
    """Create an independent YOLO YAML with verified dataset paths.

    The original dataset YAML can contain paths from an earlier
    Colab session. This function creates a temporary configuration
    with the current absolute dataset root and the same four classes.

    Args:
        dataset_root: Existing four-class dataset directory.
        dataset_config: Verified original YOLO dataset configuration.
        runtime_directory: Temporary directory for the runtime YAML.

    Returns:
        Path to the temporary YOLO configuration file.
    """
    runtime_config = dict(dataset_config)

    runtime_config["path"] = str(
        dataset_root.resolve()
    )

    runtime_config["train"] = "train/images"
    runtime_config["val"] = "valid/images"
    runtime_config["test"] = "test/images"

    runtime_config["nc"] = len(FOUR_CLASS_NAMES)
    runtime_config["names"] = list(FOUR_CLASS_NAMES)

    runtime_yaml = (
        runtime_directory / "data_runtime.yaml"
    )

    with runtime_yaml.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            runtime_config,
            file,
            sort_keys=False,
        )

    return runtime_yaml


def train_yolo(
    yaml_path: Path,
    output_dir: Path,
    device: int,
) -> Path:
    """Train YOLOv8n using the original four-class configuration.

    Training starts from the pretrained `yolov8n.pt` weights.
    The model is trained for 30 epochs using a batch size of 16,
    640-pixel images and random seed 42.

    The training procedure produces checkpoint weights, metrics,
    training-history records and Ultralytics diagnostic figures.

    Args:
        yaml_path: Runtime dataset YAML containing valid paths.
        output_dir: New directory for reproduction artifacts.
        device: CUDA device index used for model training.

    Returns:
        Path to the best checkpoint saved during the experiment.

    Raises:
        RuntimeError: If the expected best-model checkpoint
            was not created by the training procedure.
    """
    model = YOLO(INITIAL_WEIGHTS)

    print("\nYOLOV8N FOUR-CLASS TRAINING")
    print("Initial weights:", INITIAL_WEIGHTS)
    print("Epochs:", EPOCHS)
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Workers:", WORKERS)
    print("Random seed:", SEED)
    print("Device:", device)
    print("Output:", output_dir)

    # Preserve the original Ultralytics training settings.
    model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        seed=SEED,
        workers=WORKERS,
        device=device,
        project=str(output_dir.parent),
        name=output_dir.name,
        plots=True,
        exist_ok=False,
    )

    best_checkpoint = (
        output_dir
        / "weights"
        / "best.pt"
    )

    if not best_checkpoint.is_file():
        raise RuntimeError(
            "Training finished without the expected "
            f"best checkpoint: {best_checkpoint}"
        )

    print("\nTRAINING COMPLETED")
    print("Best checkpoint:", best_checkpoint)
    print("Training history:", output_dir / "results.csv")

    return best_checkpoint


def evaluate_yolo(
    checkpoint_path: Path,
    yaml_path: Path,
    output_dir: Path,
    device: int,
) -> dict:
    """Evaluate the newly trained YOLOv8n model on validation data.

    The best saved checkpoint is evaluated with the same 640-pixel
    inference size used during training. Overall and per-class
    average precision are recorded for reproducibility.

    Args:
        checkpoint_path: Path to the newly saved best YOLO weights.
        yaml_path: Dataset YAML used for validation.
        output_dir: Reproduction directory for evaluation artifacts.
        device: CUDA device index used for model inference.

    Returns:
        Dictionary containing precision, recall, mAP@50,
        mAP@50:95 and per-class AP@50:95.
    """
    model = YOLO(
        str(checkpoint_path)
    )

    # Run the same validation procedure as the cleaned Notebook 04.
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
        verbose=False,
    )

    per_class_ap = {
        str(model.names[class_id]): float(ap)
        for class_id, ap in enumerate(metrics.box.maps)
    }

    results = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "per_class_mAP50_95": per_class_ap,
    }

    print("\nVALIDATION RESULTS")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"mAP@50: {results['mAP50']:.4f}")
    print(f"mAP@50:95: {results['mAP50_95']:.4f}")

    print("\nPER-CLASS AP@50:95")

    for class_name, ap in per_class_ap.items():
        print(f"{class_name}: {ap:.4f}")

    metrics_path = (
        output_dir / "validation_metrics.json"
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print("Metrics saved:", metrics_path)

    return results


def main() -> None:
    """Execute the four-class YOLOv8n training reproduction.

    The original four-class dataset is verified and a temporary
    runtime YAML is created to correct dataset paths without
    changing the original dataset configuration.

    Training and validation artifacts are written to a separate
    output directory to protect the original model checkpoint.

    Args:
        None. Project and output paths are supplied through
        command-line arguments.

    Returns:
        None. The function trains the model and saves checkpoint
        weights, training figures and validation results.

    Raises:
        RuntimeError: If CUDA or the required GPU device
            is unavailable.
        FileNotFoundError: If the original dataset is missing.
        ValueError: If the four-class mapping is incorrect or
            a protected directory is selected.
        FileExistsError: If the reproduction directory exists.
    """
    args = parse_args()

    set_seed(SEED)

    project_root = (
        args.project_root.expanduser().resolve()
    )

    dataset_root = get_dataset_path(
        project_root,
        four_class=True,
    )

    dataset_config = verify_dataset(
        dataset_root
    )

    output_dir = prepare_output_directory(
        project_root=project_root,
        custom_output_dir=args.output_dir,
    )

    # Use GPU device zero to reproduce the original Colab setup.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the original YOLO training setup."
        )

    device = 0

    if torch.cuda.device_count() <= device:
        raise RuntimeError(
            f"CUDA device {device} is unavailable."
        )

    print("\nFOUR-CLASS YOLO REPRODUCTION")
    print("Dataset:", dataset_root)
    print("Output directory:", output_dir)
    print("Device:", device)

    # Use a temporary configuration instead of editing data.yaml.
    with tempfile.TemporaryDirectory() as temporary_folder:
        temporary_directory = Path(temporary_folder)

        runtime_yaml = create_runtime_yaml(
            dataset_root=dataset_root,
            dataset_config=dataset_config,
            runtime_directory=temporary_directory,
        )

        best_checkpoint = train_yolo(
            yaml_path=runtime_yaml,
            output_dir=output_dir,
            device=device,
        )

        evaluate_yolo(
            checkpoint_path=best_checkpoint,
            yaml_path=runtime_yaml,
            output_dir=output_dir,
            device=device,
        )

    print("\nREPRODUCTION COMPLETED")
    print("Run directory:", output_dir)
    print("Best checkpoint:", best_checkpoint)
    print(
        "Validation metrics:",
        output_dir / "validation_metrics.json",
    )


if __name__ == "__main__":
    main()
