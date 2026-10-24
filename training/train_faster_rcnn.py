
# Reproduce the original four-class Faster R-CNN training experiment.

import argparse
import sys
import time

from pathlib import Path

import pandas as pd
import torch

from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# Import the shared project modules.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mma3001.config import get_dataset_path, set_seed
from mma3001.dataset import (
    YoloDetectionDataset,
    detection_collate_fn,
)


# Original experiment configuration.
SEED = 42
NUM_CLASSES = 4
NUM_EPOCHS = 15
BATCH_SIZE = 4

LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

STEP_SIZE = 5
GAMMA = 0.1
IMAGE_SIZE = 640


def parse_args():
    """Parse command-line arguments for the training experiment.

    Returns:
        Namespace containing the project root and reproduction
        output directory.
    """
    parser = argparse.ArgumentParser(
        description="Reproduce four-class Faster R-CNN training."
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
        help="Optional directory for new reproduction outputs.",
    )

    return parser.parse_args()


def create_dataloaders(dataset_root: Path, device: torch.device):
    """Create the original training and validation data loaders.

    Args:
        dataset_root: Directory containing the four-class dataset.
        device: PyTorch device used during model training.

    Returns:
        Tuple containing the training and validation DataLoaders.

    Raises:
        ValueError: If the dataset image counts differ from the
            original four-class experiment.
    """
    train_dataset = YoloDetectionDataset(
        dataset_root=dataset_root,
        split="train",
        label_offset=1,
    )

    valid_dataset = YoloDetectionDataset(
        dataset_root=dataset_root,
        split="valid",
        label_offset=1,
    )

    if len(train_dataset) != 3138 or len(valid_dataset) != 120:
        raise ValueError(
            "Dataset image counts differ from the original experiment. "
            f"Train={len(train_dataset)}, "
            f"Validation={len(valid_dataset)}"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        collate_fn=detection_collate_fn,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        collate_fn=detection_collate_fn,
    )

    return train_loader, valid_loader


def create_model(device: torch.device) -> torch.nn.Module:
    """Initialise the original COCO-pretrained Faster R-CNN architecture.

    The pretrained classification head is replaced with a new predictor
    containing four foreground defect classes and one background class.
    The original image-size and detection postprocessing settings
    are retained.

    Args:
        device: PyTorch device on which the model will be trained.

    Returns:
        Faster R-CNN ResNet50-FPN V2 model configured for four
        defect categories and transferred to the requested device.
    """
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        min_size=IMAGE_SIZE,
        max_size=IMAGE_SIZE,
    )

    in_features = (
        model.roi_heads.box_predictor.cls_score.in_features
    )

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_channels=in_features,
        num_classes=NUM_CLASSES + 1,
    )

    # Restore the original inference postprocessing settings.
    model.roi_heads.score_thresh = 0.001
    model.roi_heads.nms_thresh = 0.50
    model.roi_heads.detections_per_img = 100

    return model.to(device)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate the detector using COCO-style validation metrics.

    Model predictions and ground-truth annotations are transferred
    to the CPU before updating the TorchMetrics evaluator.

    Args:
        model: Faster R-CNN model to evaluate.
        data_loader: DataLoader providing validation images and
            ground-truth detection targets.
        device: PyTorch device used for model inference.

    Returns:
        Dictionary containing COCO-style evaluation metrics,
        including mAP@50, mAP@50:95 and per-class AP.
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

        metric.update(predictions_cpu, targets_cpu)

    results = metric.compute()
    metric.reset()

    return results


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
) -> None:
    """Train Faster R-CNN and save the best validation checkpoint.

    The model is trained for 15 epochs using SGD, StepLR scheduling
    and CUDA mixed precision when available. COCO-style validation
    metrics are computed after each epoch.

    The checkpoint is updated whenever validation mAP@50:95 improves.
    Training history is exported after every epoch.

    Args:
        model: Initialised Faster R-CNN model.
        train_loader: DataLoader containing training images and targets.
        valid_loader: DataLoader containing validation images and targets.
        device: PyTorch device used for training and validation.
        output_dir: Directory for saving the best checkpoint and
            training-history CSV file.

    Returns:
        None. The function saves `best.pt` and
        `training_history.csv` to the output directory.
    """
    optimizer = torch.optim.SGD(
        model.parameters(),
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

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    history = []
    best_map = -1.0
    best_epoch = None

    start_training = time.perf_counter()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()

        epoch_loss = 0.0
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

            # Use mixed precision on CUDA.
            with torch.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                loss_dict = model(
                    images_device,
                    targets_device,
                )

                loss = sum(loss_dict.values())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.item())

            if batch_index % 50 == 0:
                print(
                    f"Epoch {epoch}/{NUM_EPOCHS} "
                    f"Batch {batch_index}/{len(train_loader)} "
                    f"Loss: {loss.item():.4f}"
                )

        average_loss = epoch_loss / len(train_loader)

        # Evaluate on validation data before saving the checkpoint.
        validation_metrics = evaluate_model(
            model,
            valid_loader,
            device,
        )

        validation_map50 = float(
            validation_metrics["map_50"]
        )

        validation_map = float(
            validation_metrics["map"]
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

        # Preserve the original training-history column format.
        history_df = pd.DataFrame(history)

        history_df.to_csv(
            output_dir / "training_history.csv",
            index=False,
        )

        if validation_map > best_map:
            best_map = validation_map
            best_epoch = epoch

            # Preserve the original checkpoint structure.
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_map": validation_map,
            }

            torch.save(
                checkpoint,
                output_dir / "best.pt",
            )

            print(
                f"New best checkpoint saved at epoch {epoch}."
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
        time.perf_counter() - start_training
    )

    print("\nTRAINING COMPLETED")
    print("Total epochs:", NUM_EPOCHS)
    print("Best epoch:", best_epoch)
    print("Best validation mAP@50:95:", round(best_map, 4))
    print("Total training duration:", round(total_seconds, 2), "s")

    print(
        "Best checkpoint:",
        output_dir / "best.pt",
    )

    print(
        "Training history:",
        output_dir / "training_history.csv",
    )


def main() -> None:
    """Configure and execute the Faster R-CNN reproduction experiment.

    The project dataset is loaded from the specified project directory.
    A separate reproduction directory is created to protect the
    original checkpoint and training results.

    Args:
        None. Configuration is provided through command-line arguments.

    Returns:
        None. The training procedure writes a checkpoint and
        training-history CSV to the reproduction directory.

    Raises:
        ValueError: If the selected output directory is the original
            Faster R-CNN experiment directory.
        FileExistsError: If the reproduction output directory
            already exists.
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

    # Protect the original trained checkpoint.
    original_run = (
        project_root
        / "model_runs"
        / "faster_rcnn_resnet50_4class"
    ).resolve()

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else (
            project_root
            / "model_runs"
            / "reproduction_faster_rcnn_4class"
        )
    ).expanduser().resolve()

    if output_dir == original_run:
        raise ValueError(
            "Refusing to overwrite the original Faster R-CNN run."
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

    print("FASTER R-CNN FOUR-CLASS TRAINING")
    print("Device:", device)
    print("Dataset:", dataset_root)
    print("Output directory:", output_dir)
    print("Random seed:", SEED)
    print("Epochs:", NUM_EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)

    train_loader, valid_loader = create_dataloaders(
        dataset_root,
        device,
    )

    model = create_model(device)

    train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
