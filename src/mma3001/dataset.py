
"""Dataset loading and annotation conversion for MMA3001 experiments.

This module reads YOLO-format object detection datasets and provides
Torchvision-compatible datasets for RetinaNet and Faster R-CNN.

The functions do not modify source images or annotation files.
"""

from pathlib import Path
from typing import Callable

import numpy as np
import torch
import yaml

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_class_names(dataset_root: str | Path) -> dict[int, str]:
    """Read class IDs and names from a YOLO dataset configuration.

    Args:
        dataset_root: Directory containing data.yaml.

    Returns:
        Dictionary mapping zero-based class IDs to class names.

    Raises:
        FileNotFoundError: If data.yaml is missing.
        ValueError: If the class mapping is invalid.
    """
    yaml_path = Path(dataset_root) / "data.yaml"

    if not yaml_path.is_file():
        raise FileNotFoundError(
            f"Dataset configuration not found: {yaml_path}"
        )

    with yaml_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict) or "names" not in config:
        raise ValueError(
            f"Missing class names in {yaml_path}"
        )

    names = config["names"]

    # Handle the dictionary and list formats supported by YOLO.
    if isinstance(names, dict):
        class_names = {
            int(index): str(name)
            for index, name in names.items()
        }
    elif isinstance(names, list):
        class_names = dict(enumerate(map(str, names)))
    else:
        raise ValueError("Class names must be a dictionary or list.")

    # The class IDs must be consecutive and start from zero.
    if (
        not class_names
        or sorted(class_names) != list(range(len(class_names)))
    ):
        raise ValueError(
            "YOLO class IDs must be consecutive and zero-based."
        )

    return class_names


def list_images(
    dataset_root: str | Path,
    split: str,
) -> list[Path]:
    """Return image paths for a dataset split in sorted order.

    Args:
        dataset_root: Root directory of the dataset.
        split: Dataset split, such as train, valid or test.

    Returns:
        List of paths to supported image files.

    Raises:
        FileNotFoundError: If the image directory is missing.
    """
    images_dir = Path(dataset_root) / split / "images"

    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {images_dir}"
        )

    # Sorting ensures a consistent image order across evaluations.
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_yolo_annotations(
    label_path: str | Path,
    num_classes: int,
    allow_missing: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one YOLO annotation file without changing its contents.

    Args:
        label_path: Path to a YOLO text annotation file.
        num_classes: Number of classes defined in the dataset.
        allow_missing: Whether a missing label file represents background.

    Returns:
        Tuple containing:
            - Class IDs with shape (N,).
            - Normalised bounding boxes with shape (N, 4).
              Coordinates use x_center, y_center, width, height.

    Raises:
        FileNotFoundError: If a label is missing and not allowed.
        ValueError: If any annotation is malformed or invalid.
    """
    label_path = Path(label_path)

    if not label_path.is_file():
        if not allow_missing:
            raise FileNotFoundError(
                f"Annotation file not found: {label_path}"
            )

        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0, 4), dtype=np.float32),
        )

    text = label_path.read_text(encoding="utf-8").strip()

    # Empty labels are treated as background-only images.
    if not text:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0, 4), dtype=np.float32),
        )

    labels = []
    boxes = []

    for line_number, line in enumerate(
        text.splitlines(), start=1
    ):
        values = line.split()

        # YOLO detection annotations require exactly five values.
        if len(values) != 5:
            raise ValueError(
                f"{label_path}, line {line_number}: "
                "expected five annotation values."
            )

        try:
            class_value = float(values[0])
            coordinates = np.asarray(
                [float(value) for value in values[1:]],
                dtype=np.float64,
            )
        except ValueError as error:
            raise ValueError(
                f"{label_path}, line {line_number}: "
                "invalid numeric annotation."
            ) from error

        # Reject NaN, infinity and non-integer class IDs.
        if (
            not np.isfinite(class_value)
            or not class_value.is_integer()
            or not np.all(np.isfinite(coordinates))
        ):
            raise ValueError(
                f"{label_path}, line {line_number}: "
                "non-finite values or invalid class ID."
            )

        class_id = int(class_value)

        if not 0 <= class_id < num_classes:
            raise ValueError(
                f"{label_path}, line {line_number}: "
                f"class {class_id} is outside the valid range."
            )

        xc, yc, width, height = coordinates

        # Check normalised centre coordinates and box dimensions.
        if not (
            0 <= xc <= 1
            and 0 <= yc <= 1
            and 0 < width <= 1
            and 0 < height <= 1
        ):
            raise ValueError(
                f"{label_path}, line {line_number}: "
                "invalid normalised bounding box."
            )

        labels.append(class_id)
        boxes.append(coordinates)

    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
    )


def yolo_to_xyxy(
    boxes: np.ndarray,
    image_width: int,
    image_height: int,
    clip: bool = True,
) -> np.ndarray:
    """Convert normalised YOLO boxes into pixel-coordinate XYXY boxes.

    Args:
        boxes: Array of YOLO boxes with shape (N, 4).
        image_width: Original image width in pixels.
        image_height: Original image height in pixels.
        clip: Whether to clip box coordinates to image boundaries.

    Returns:
        Array of pixel-coordinate boxes with shape (N, 4).

    Raises:
        ValueError: If the dimensions or bounding boxes are invalid.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")

    boxes = np.asarray(
        boxes, dtype=np.float32
    ).reshape(-1, 4)

    if not np.all(np.isfinite(boxes)):
        raise ValueError("Bounding boxes contain non-finite values.")

    converted = np.empty_like(boxes)

    # Convert each normalised centre and size into pixel corners.
    converted[:, 0] = (
        boxes[:, 0] - boxes[:, 2] / 2
    ) * image_width

    converted[:, 1] = (
        boxes[:, 1] - boxes[:, 3] / 2
    ) * image_height

    converted[:, 2] = (
        boxes[:, 0] + boxes[:, 2] / 2
    ) * image_width

    converted[:, 3] = (
        boxes[:, 1] + boxes[:, 3] / 2
    ) * image_height

    if clip:
        converted[:, [0, 2]] = np.clip(
            converted[:, [0, 2]], 0, image_width
        )
        converted[:, [1, 3]] = np.clip(
            converted[:, [1, 3]], 0, image_height
        )

    # Degenerate boxes cannot be used as detection targets.
    if np.any(
        (converted[:, 2] <= converted[:, 0])
        | (converted[:, 3] <= converted[:, 1])
    ):
        raise ValueError(
            "Bounding-box conversion produced a zero-area box."
        )

    return converted


def load_detection_target(
    label_path: str | Path,
    image_width: int,
    image_height: int,
    num_classes: int,
    image_id: int = 0,
    label_offset: int = 0,
    allow_missing: bool = True,
) -> dict[str, torch.Tensor]:
    """Build a Torchvision-style target from YOLO annotations.

    Args:
        label_path: Annotation text-file path.
        image_width: Original image width in pixels.
        image_height: Original image height in pixels.
        num_classes: Number of defect classes.
        image_id: Numeric image identifier.
        label_offset: Offset added to original YOLO class IDs.
            Use 1 for the original RetinaNet/Faster R-CNN mapping
            and 0 for the original YOLO mapping.
        allow_missing: Whether a missing annotation represents background.

    Returns:
        Dictionary containing boxes, labels, image_id, area and iscrowd.
    """
    labels, yolo_boxes = read_yolo_annotations(
        label_path,
        num_classes=num_classes,
        allow_missing=allow_missing,
    )

    pixel_boxes = yolo_to_xyxy(
        yolo_boxes,
        image_width,
        image_height,
    )

    boxes_tensor = torch.from_numpy(
        pixel_boxes.copy()
    ).float()

    labels_tensor = torch.from_numpy(
        labels.copy()
    ).long() + label_offset

    # Calculate box areas in original image pixels.
    areas = (
        (boxes_tensor[:, 2] - boxes_tensor[:, 0])
        * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
    )

    return {
        "boxes": boxes_tensor,
        "labels": labels_tensor,
        "image_id": torch.tensor(
            image_id, dtype=torch.int64
        ),
        "area": areas,
        "iscrowd": torch.zeros(
            len(labels_tensor), dtype=torch.int64
        ),
    }


class YoloDetectionDataset(Dataset):
    """Load a YOLO dataset for Torchvision object detection models.

    The dataset does not modify image or label files.

    Args:
        dataset_root: Dataset directory containing data.yaml.
        split: Dataset split, such as train, valid or test.
        label_offset: Class-ID offset applied to YOLO annotations.
            Use 1 for the original RetinaNet/Faster R-CNN training
            conventions and 0 for YOLO-compatible evaluation.
        transforms: Optional callable accepting an image tensor
            and target dictionary and returning the transformed pair.
        allow_missing_labels: Treat missing label files as background
            images when True.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        label_offset: int = 1,
        transforms: Callable | None = None,
        allow_missing_labels: bool = True,
    ):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.label_offset = label_offset
        self.transforms = transforms
        self.allow_missing_labels = allow_missing_labels

        self.class_names = load_class_names(
            self.dataset_root
        )
        self.num_classes = len(self.class_names)

        # Preserve consistent dataset ordering across evaluations.
        self.image_paths = list_images(
            self.dataset_root, split
        )

        self.labels_dir = (
            self.dataset_root / split / "labels"
        )

        if not self.labels_dir.is_dir():
            raise FileNotFoundError(
                f"Label directory not found: {self.labels_dir}"
            )

    def __len__(self) -> int:
        """Return the number of images in the split."""
        return len(self.image_paths)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Load an image and its converted detection target.

        Args:
            index: Index of the image to load.

        Returns:
            Image tensor with shape (3, H, W) and target dictionary.
        """
        image_path = self.image_paths[index]
        label_path = (
            self.labels_dir / f"{image_path.stem}.txt"
        )

        # Read the source image in RGB without modifying it.
        with Image.open(image_path) as source:
            image = source.convert("RGB")

        image_width, image_height = image.size

        target = load_detection_target(
            label_path=label_path,
            image_width=image_width,
            image_height=image_height,
            num_classes=self.num_classes,
            image_id=index,
            label_offset=self.label_offset,
            allow_missing=self.allow_missing_labels,
        )

        image_tensor = F.pil_to_tensor(
            image
        ).float() / 255.0

        if self.transforms is not None:
            image_tensor, target = self.transforms(
                image_tensor, target
            )

        return image_tensor, target


def detection_collate_fn(batch):
    """Collate variable-length object detection targets.

    Args:
        batch: Sequence of image and target pairs.

    Returns:
        Tuple containing image and target sequences.
    """
    return tuple(zip(*batch))
