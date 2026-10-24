
"""Unit tests for YOLO dataset loading and annotation conversion."""

from pathlib import Path

from PIL import Image
from mma3001.dataset import YoloDetectionDataset


def test_dataset_loading(tmp_path: Path) -> None:
    """Verify image loading and YOLO-to-Torchvision conversion."""

    # Create an artificial four-class dataset for testing.
    root = tmp_path
    images = root / "valid" / "images"
    labels = root / "valid" / "labels"

    images.mkdir(parents=True)
    labels.mkdir(parents=True)

    # Define the same four defect classes used in the final project.
    (root / "data.yaml").write_text(
        "names: [loose-meat, packaging-error, twisted-meat, unsealed]\n",
        encoding="utf-8"
    )

    # Create an image with known dimensions and bounding-box coordinates.
    Image.new("RGB", (1000, 500)).save(images / "sample.jpg")

    (labels / "sample.txt").write_text(
        "0 0.5 0.5 0.2 0.4\n",
        encoding="utf-8"
    )

    # Use one-based labels to match the Torchvision experiment convention.
    dataset = YoloDetectionDataset(
        root,
        "valid",
        label_offset=1
    )

    image, target = dataset[0]

    # Verify image dimensions and converted bounding-box coordinates.
    assert len(dataset) == 1
    assert image.shape == (3, 500, 1000)

    assert target["boxes"].tolist() == [
        [400.0, 150.0, 600.0, 350.0]
    ]

    # YOLO class 0 becomes Torchvision class 1.
    assert target["labels"].tolist() == [1]
