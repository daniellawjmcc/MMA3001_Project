
# Reproduce the original five-class and four-class RetinaNet experiments.

import argparse
import sys
import time

from functools import partial
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import (
    RetinaNet_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2,
)
from torchvision.models.detection.retinanet import (
    RetinaNetClassificationHead,
)


# Import the shared MMA3001 project modules.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mma3001.config import (
    FOUR_CLASS_NAMES,
    FIVE_CLASS_NAMES,
    SEED,
    get_dataset_path,
    set_seed,
)
from mma3001.dataset import (
    YoloDetectionDataset,
    detection_collate_fn,
)


# Original training hyperparameters.
NUM_EPOCHS = 15
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
STEP_SIZE = 5
GAMMA = 0.1

# Configuration differences between the original experiments.
EXPERIMENTS = {
    "5class": {
        "num_classes": 5,
        "class_names": FIVE_CLASS_NAMES,
        "batch_size": 8,
        "num_workers": 0,
        "image_size": None,
        "checkpoint_format": "state_dict",
        "original_run": "retinanet_resnet50_v1",
        "reproduction_run": "reproduction_retinanet_5class",
        "train_images": None,
        "valid_images": None,
    },
    "4class": {
        "num_classes": 4,
        "class_names": FOUR_CLASS_NAMES,
        "batch_size": 4,
        "num_workers": 2,
        "image_size": 640,
        "checkpoint_format": "full",
        "original_run": "retinanet_resnet50_4class",
        "reproduction_run": "reproduction_retinanet_4class",
        "train_images": 3138,
        "valid_images": 120,
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for RetinaNet training.

    Returns:
        Namespace containing the experiment type, project directory
        and optional reproduction output directory.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the original MMA3001 RetinaNet "
            "object detection experiments."
        )
    )

    parser.add_argument(
        "--experiment",
        type=str,
        choices=["5class", "4class"],
        required=True,
        help="Select the original five-class or final four-class run.",
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="Project directory containing datasets and model runs.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional new directory for reproduction outputs.",
    )

    return parser.parse_args()


def create_dataloaders(
    dataset_root: Path,
    config: dict,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation loaders for the selected experiment.

    The shared dataset loader converts normalised YOLO bounding boxes
    to absolute pixel coordinates and shifts class labels to the
    one-based class convention used in the original experiments.

    Args:
        dataset_root: Directory containing the selected dataset.
        config: Experiment configuration with class count, batch size
            and expected dataset image counts.
        device: PyTorch device used during training.

    Returns:
        Tuple containing the training and validation DataLoaders.

    Raises:
        ValueError: If the dataset image counts do not match the
            expected counts recorded for the experiment.
    """
    train_dataset = YoloDetectionDataset(
        dataset_root,
        split="train",
        label_offset=1,
    )

    valid_dataset = YoloDetectionDataset(
        dataset_root,
        split="valid",
        label_offset=1,
    )

    expected_train = config["train_images"]
    expected_valid = config["valid_images"]

    if expected_train is not None:
        if len(train_dataset) != expected_train:
            raise ValueError(
                "Unexpected training image count: "
                f"{len(train_dataset)} instead of {expected_train}."
            )

    if expected_valid is not None:
        if len(valid_dataset) != expected_valid:
            raise ValueError(
                "Unexpected validation image count: "
                f"{len(valid_dataset)} instead of {expected_valid}."
            )

    print("Training images:", len(train_dataset))
    print("Validation images:", len(valid_dataset))

    batch_size = config["batch_size"]
    num_workers = config["num_workers"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=detection_collate_fn,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=detection_collate_fn,
    )

    return train_loader, valid_loader


def create_model(
    config: dict,
    device: torch.device,
) -> torch.nn.Module:
    """Initialise the original COCO-pretrained RetinaNet architecture.

    The model uses a ResNet50-FPN V2 backbone with a replacement
    GroupNorm classification head. The head supports the selected
    number of defect classes plus the reserved background index.

    The four-class experiment uses 640-pixel image resizing. The
    five-class experiment retains the original Torchvision default.

    Args:
        config: Experiment configuration containing class count
            and image-resizing settings.
        device: PyTorch device on which the model will run.

    Returns:
        Configured RetinaNet ResNet50-FPN V2 model transferred
        to the requested device.
    """
    weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT

    model_kwargs = {
        "weights": weights,
    }

    image_size = config["image_size"]

    if image_size is not None:
        model_kwargs["min_size"] = image_size
        model_kwargs["max_size"] = image_size

    model = retinanet_resnet50_fpn_v2(
        **model_kwargs
    )

    in_channels = model.backbone.out_channels

    num_anchors = (
        model.anchor_generator
        .num_anchors_per_location()[0]
    )

    num_model_classes = config["num_classes"] + 1

    # Replace the COCO classification head with the experiment head.
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_model_classes,
        norm_layer=partial(nn.GroupNorm, 32),
    )

    model.detections_per_img = 100

    model = model.to(device)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nRETINANET MODEL")
    print("Architecture: RetinaNet ResNet50-FPN V2")
    print("Pretrained weights: COCO")
    print("Foreground classes:", config["num_classes"])
    print("Classification outputs:", num_model_classes)
    print("Total parameters:", total_parameters)
    print("Trainable parameters:", trainable_parameters)

    print(
        "Image size:",
        image_size if image_size is not None
        else "Torchvision default",
    )

    return model


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Calculate COCO-style detection metrics on the validation split.

    The model performs inference without updating its parameters.
    Predictions and ground-truth annotations are transferred to
    the CPU before being passed to TorchMetrics.

    Args:
        model: Trained RetinaNet model to evaluate.
        data_loader: Validation DataLoader containing images and
            ground-truth detection targets.
        device: PyTorch device used for model inference.

    Returns:
        Dictionary of COCO-style detection metrics, including
        mAP@50, mAP@50:95, mAP@75 and mean average recall.
    """
    model.eval()

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=True,
    )

    for images, targets in data_loader:
        images_device = [
            image.to(device)
            for image in images
        ]

        predictions = model(images_device)

        predictions_cpu = [
            {
                "boxes": prediction["boxes"].detach().cpu(),
                "scores": prediction["scores"].detach().cpu(),
                "labels": prediction["labels"].detach().cpu(),
            }
            for prediction in predictions
        ]

        targets_cpu = [
            {
                "boxes": target["boxes"].detach().cpu(),
                "labels": target["labels"].detach().cpu(),
            }
            for target in targets
        ]

        metric.update(
            predictions_cpu,
            targets_cpu,
        )

    results = metric.compute()
    metric.reset()

    return results


def save_best_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_map: float,
    output_dir: Path,
    checkpoint_format: str,
) -> None:
    """Save the best model using the selected original checkpoint format.

    The preliminary five-class experiment uses a raw model state
    dictionary. The final four-class experiment stores model
    weights, optimiser state, epoch and validation mAP.

    Args:
        model: RetinaNet model whose weights will be saved.
        optimizer: Optimiser associated with the current training run.
        epoch: One-based training epoch of the checkpoint.
        validation_map: Validation mAP@50:95 obtained at this epoch.
        output_dir: Directory used to save reproduction artifacts.
        checkpoint_format: Either 'state_dict' or 'full'.

    Returns:
        None. The checkpoint is written to `best.pt`.

    Raises:
        ValueError: If the requested checkpoint format is unsupported.
    """
    checkpoint_path = output_dir / "best.pt"

    if checkpoint_format == "state_dict":
        checkpoint = model.state_dict()

    elif checkpoint_format == "full":
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "val_map": validation_map,
        }

    else:
        raise ValueError(
            f"Unsupported checkpoint format: {checkpoint_format}"
        )

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    print(
        f"New best checkpoint saved at epoch {epoch}: "
        f"{checkpoint_path}"
    )


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    config: dict,
    device: torch.device,
    output_dir: Path,
) -> None:
    """Train RetinaNet and select the best validation checkpoint.

    The model is fine-tuned for 15 epochs using SGD, StepLR
    scheduling and mixed precision when CUDA is available.

    COCO-style metrics are calculated after each epoch, and
    the checkpoint is updated whenever validation mAP@50:95
    improves.

    Training history is saved after each epoch. The CSV format
    follows the original four-class experiment and is also
    exported for the preliminary five-class experiment.

    Args:
        model: Initialised RetinaNet model.
        train_loader: DataLoader providing training images and targets.
        valid_loader: DataLoader providing validation images and targets.
        config: Experiment configuration containing checkpoint format
            and other experiment-specific settings.
        device: PyTorch device used for training and validation.
        output_dir: Directory for saved checkpoints and training logs.

    Returns:
        None. Saves `best.pt` and `training_history.csv` to the
        selected output directory.
    """
    optimizer = torch.optim.SGD(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=STEP_SIZE,
        gamma=GAMMA,
    )

    amp_enabled = device.type == "cuda"

    # Use one gradient scaler for the complete training run.
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    history = []
    best_map = -1.0
    best_epoch = None

    training_start = time.perf_counter()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()

        running_loss = 0.0
        epoch_start = time.perf_counter()

        for batch_index, (images, targets) in enumerate(
            train_loader,
            start=1,
        ):
            images_device = [
                image.to(device, non_blocking=True)
                for image in images
            ]

            targets_device = [
                {
                    key: value.to(device, non_blocking=True)
                    for key, value in target.items()
                }
                for target in targets
            ]

            optimizer.zero_grad(set_to_none=True)

            # Forward pass using CUDA mixed precision when available.
            with torch.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                loss_dict = model(
                    images_device,
                    targets_device,
                )

                total_loss = sum(
                    loss_dict.values()
                )

            # Calculate gradients and update model parameters.
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(
                total_loss.item()
            )

            print_interval = (
                20 if config["num_classes"] == 5
                else 50
            )

            if batch_index % print_interval == 0:
                print(
                    f"Epoch {epoch}/{NUM_EPOCHS} | "
                    f"Batch {batch_index}/{len(train_loader)} | "
                    f"Loss {total_loss.item():.4f}"
                )

        average_loss = (
            running_loss / len(train_loader)
        )

        # Evaluate before deciding whether to save a new checkpoint.
        validation_results = evaluate_model(
            model,
            valid_loader,
            device,
        )

        validation_map = float(
            validation_results["map"]
        )

        validation_map50 = float(
            validation_results["map_50"]
        )

        epoch_seconds = (
            time.perf_counter() - epoch_start
        )

        history.append({
            "Epoch": epoch,
            "Train_Loss": average_loss,
            "Val_mAP50": validation_map50,
            "Val_mAP50_95": validation_map,
            "Epoch_Seconds": epoch_seconds,
        })

        # Save the measured training and validation history.
        history_df = pd.DataFrame(history)

        history_df.to_csv(
            output_dir / "training_history.csv",
            index=False,
        )

        if validation_map > best_map:
            best_map = validation_map
            best_epoch = epoch

            save_best_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_map=validation_map,
                output_dir=output_dir,
                checkpoint_format=config["checkpoint_format"],
            )

        scheduler.step()

        print(
            f"\nEpoch {epoch}/{NUM_EPOCHS}"
            f"\nTraining loss: {average_loss:.4f}"
            f"\nValidation mAP@50: {validation_map50:.4f}"
            f"\nValidation mAP@50:95: {validation_map:.4f}"
            f"\nEpoch duration: {epoch_seconds:.2f} s"
            f"\nBest mAP@50:95: {best_map:.4f}\n"
        )

    total_seconds = (
        time.perf_counter() - training_start
    )

    print("\nRETINANET TRAINING COMPLETED")
    print("Total epochs:", NUM_EPOCHS)
    print("Best epoch:", best_epoch)
    print("Best validation mAP@50:95:", round(best_map, 4))
    print("Total training duration:", round(total_seconds, 2), "s")
    print("Checkpoint:", output_dir / "best.pt")
    print("History:", output_dir / "training_history.csv")


def main() -> None:
    """Configure and execute the selected RetinaNet training experiment.

    The original dataset is loaded using the shared project modules.
    All training outputs are written to a new reproduction directory
    to protect the original saved model checkpoints.

    Args:
        None. Experiment selection and project paths are provided
        through command-line arguments.

    Returns:
        None. Starts the selected training experiment and writes
        its artifacts to the reproduction output directory.

    Raises:
        ValueError: If the output directory overlaps an original
            experiment or an existing data/checkpoint location.
        FileExistsError: If the reproduction directory already exists.
        FileNotFoundError: If the selected dataset is unavailable.
    """
    args = parse_args()

    set_seed(SEED)

    config = EXPERIMENTS[args.experiment]

    project_root = (
        args.project_root.expanduser().resolve()
    )

    dataset_root = get_dataset_path(
        project_root,
        four_class=(args.experiment == "4class"),
    )

    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_root}"
        )

    original_runs = [
        (
            project_root
            / "model_runs"
            / experiment["original_run"]
        ).resolve()
        for experiment in EXPERIMENTS.values()
    ]

    if args.output_dir is None:
        output_dir = (
            project_root
            / "model_runs"
            / config["reproduction_run"]
        )
    else:
        output_dir = args.output_dir

    output_dir = output_dir.expanduser().resolve()

    # Never write training results into an existing experiment.
    protected_paths = original_runs + [
        project_root,
        dataset_root,
        project_root / "model_runs",
    ]


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
            f"Output directory already exists: {output_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("RETINANET TRAINING CONFIGURATION")
    print("Experiment:", args.experiment)
    print("Device:", device)
    print("Dataset:", dataset_root)
    print("Output directory:", output_dir)
    print("Random seed:", SEED)
    print("Defect classes:", config["class_names"])
    print("Epochs:", NUM_EPOCHS)
    print("Batch size:", config["batch_size"])
    print("Learning rate:", LEARNING_RATE)
    print("Checkpoint format:", config["checkpoint_format"])

    train_loader, valid_loader = create_dataloaders(
        dataset_root=dataset_root,
        config=config,
        device=device,
    )

    model = create_model(
        config=config,
        device=device,
    )

    train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        config=config,
        device=device,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
