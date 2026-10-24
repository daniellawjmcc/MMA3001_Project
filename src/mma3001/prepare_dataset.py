
# Reproduce the original five-class cleaning and four-class dataset preparation.

import argparse
import random
import shutil
import sys
import tempfile

from collections import Counter
from pathlib import Path

import yaml


# Import the existing dataset utilities.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mma3001.config import (
    FIVE_CLASS_NAMES,
    FOUR_CLASS_NAMES,
    SEED,
)
from mma3001.dataset import (
    IMAGE_EXTENSIONS,
    list_images,
    load_class_names,
    read_yolo_annotations,
)


SPLITS = ("train", "valid", "test")

ORIGINAL_DATASET = "pork_rasher_v4"
CLEAN_DATASET = "pork_rasher_v4_clean"
FOUR_CLASS_DATASET = "pork_rasher_v4_4class"

WRINKLE_CLASS_ID = 4
EXPECTED_WRINKLE_SOURCES = 6


def parse_args() -> argparse.Namespace:
    """Parse arguments for dataset preparation.

    Returns:
        Namespace containing the project directory and selected
        preparation stage.
    """
    parser = argparse.ArgumentParser(
        description="Reproduce the original MMA3001 dataset preparation."
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="Project directory containing the dataset folder.",
    )

    parser.add_argument(
        "--stage",
        choices=["clean-five", "make-four", "all"],
        required=True,
        help="Select five-class cleaning, four-class refinement or both.",
    )

    return parser.parse_args()


def verify_dataset(
    dataset_root: Path,
    expected_names: tuple[str, ...],
) -> None:
    """Verify dataset structure, class mapping and YOLO annotations.

    Args:
        dataset_root: Dataset directory containing data.yaml and splits.
        expected_names: Expected ordered defect class names.

    Returns:
        None. Prints the verified image and annotation counts.

    Raises:
        FileNotFoundError: If the dataset or required split is missing.
        ValueError: If the class mapping or an annotation is invalid.
    """
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_root}"
        )

    class_mapping = load_class_names(dataset_root)

    actual_names = [
        class_mapping[index]
        for index in range(len(class_mapping))
    ]

    if actual_names != list(expected_names):
        raise ValueError(
            "Unexpected dataset class mapping.\n"
            f"Expected: {list(expected_names)}\n"
            f"Found: {actual_names}"
        )

    print("\nDATASET VERIFICATION")
    print("Dataset:", dataset_root)
    print("Classes:", actual_names)

    for split in SPLITS:
        labels_dir = dataset_root / split / "labels"

        if not labels_dir.is_dir():
            raise FileNotFoundError(
                f"Label directory missing: {labels_dir}"
            )

        image_paths = list_images(
            dataset_root,
            split,
        )

        annotation_counts = Counter()
        empty_images = 0

        for image_path in image_paths:
            label_path = (
                labels_dir / f"{image_path.stem}.txt"
            )

            labels, _ = read_yolo_annotations(
                label_path,
                num_classes=len(expected_names),
                allow_missing=True,
            )

            if len(labels) == 0:
                empty_images += 1

            annotation_counts.update(
                int(label) for label in labels
            )

        print(
            f"{split}: {len(image_paths)} images, "
            f"{empty_images} background-only images"
        )

        for class_id, name in enumerate(expected_names):
            print(
                f"  {name}: {annotation_counts[class_id]} boxes"
            )


def get_source_id(filename_stem: str) -> str:
    """Extract the source identifier from a Roboflow image filename.

    Args:
        filename_stem: Image filename without its extension.

    Returns:
        Source prefix preceding the `.rf.` augmentation identifier.
        If the marker is absent, returns the original filename stem.
    """
    return filename_stem.split(".rf.", 1)[0]


def collect_wrinkle_sources(
    dataset_root: Path,
) -> dict[str, list[Path]]:
    """Identify wrinkle-labelled training images by source group.

    Args:
        dataset_root: Original five-class dataset directory.

    Returns:
        Dictionary mapping each unique wrinkle source identifier
        to its corresponding training image paths.

    Raises:
        ValueError: If wrinkle images are found outside training
            or the wrinkle source count differs from the original audit.
    """
    wrinkle_sources = {}

    for split in SPLITS:
        labels_dir = dataset_root / split / "labels"

        for image_path in list_images(dataset_root, split):
            label_path = labels_dir / f"{image_path.stem}.txt"

            labels, _ = read_yolo_annotations(
                label_path,
                num_classes=len(FIVE_CLASS_NAMES),
                allow_missing=True,
            )

            if WRINKLE_CLASS_ID not in labels:
                continue

            if split != "train":
                raise ValueError(
                    "The original cleaning procedure expects wrinkle "
                    f"annotations only in training: {image_path}"
                )

            source_id = get_source_id(image_path.stem)

            wrinkle_sources.setdefault(
                source_id,
                [],
            ).append(image_path)

    if len(wrinkle_sources) != EXPECTED_WRINKLE_SOURCES:
        raise ValueError(
            "Unexpected number of wrinkle source groups. "
            f"Expected {EXPECTED_WRINKLE_SOURCES}, "
            f"found {len(wrinkle_sources)}."
        )

    print("\nWRINKLE SOURCE AUDIT")
    print("Unique wrinkle sources:", len(wrinkle_sources))

    for source_id, images in sorted(wrinkle_sources.items()):
        print(f"{source_id}: {len(images)} labelled images")

    return wrinkle_sources


def write_dataset_yaml(
    dataset_root: Path,
    class_names: tuple[str, ...],
) -> None:
    """Write the YOLO dataset configuration for a prepared dataset.

    Args:
        dataset_root: Newly prepared output dataset directory.
        class_names: Ordered names of the retained defect classes.

    Returns:
        None. Writes data.yaml inside the output dataset.
    """
    config = {
        "path": str(dataset_root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": list(class_names),
    }

    yaml_path = dataset_root / "data.yaml"

    with yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
        )

    print("Dataset YAML saved:", yaml_path)


def move_source_group(
    dataset_root: Path,
    source_id: str,
    destination_split: str,
) -> int:
    """Move all training augmentations of one source into another split.

    Args:
        dataset_root: Working copy of the five-class dataset.
        source_id: Original image source identifier.
        destination_split: Destination split, either valid or test.

    Returns:
        Number of image–annotation pairs moved.

    Raises:
        ValueError: If the destination split is unsupported.
        FileNotFoundError: If a matching annotation is missing.
        FileExistsError: If a destination filename already exists.
    """
    if destination_split not in ("valid", "test"):
        raise ValueError(
            f"Invalid destination split: {destination_split}"
        )

    source_images = dataset_root / "train" / "images"
    source_labels = dataset_root / "train" / "labels"

    destination_images = (
        dataset_root / destination_split / "images"
    )
    destination_labels = (
        dataset_root / destination_split / "labels"
    )

    matching_images = sorted(
        path
        for path in source_images.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and get_source_id(path.stem) == source_id
    )

    for image_path in matching_images:
        label_path = source_labels / f"{image_path.stem}.txt"

        if not label_path.is_file():
            raise FileNotFoundError(
                f"Missing annotation for source group: {image_path}"
            )

        new_image_path = destination_images / image_path.name
        new_label_path = destination_labels / label_path.name

        if new_image_path.exists() or new_label_path.exists():
            raise FileExistsError(
                f"Destination file already exists for {image_path.name}"
            )

    for image_path in matching_images:
        label_path = source_labels / f"{image_path.stem}.txt"

        shutil.move(
            str(image_path),
            str(destination_images / image_path.name),
        )

        shutil.move(
            str(label_path),
            str(destination_labels / label_path.name),
        )

    print(
        f"Moved {source_id} to {destination_split}: "
        f"{len(matching_images)} images"
    )

    return len(matching_images)


def prepare_five_class(
    original_root: Path,
    output_root: Path,
) -> None:
    """Create the cleaned five-class dataset with source-level splits.

    Six independent wrinkle sources are identified from the
    original training set. With seed 42, one source is selected
    for validation and another for testing. The remaining four
    sources remain in training.

    Every augmented image belonging to a selected source is moved
    together with its annotation to avoid source-group leakage.

    Args:
        original_root: Original unmodified five-class dataset.
        output_root: New cleaned five-class output directory.

    Returns:
        None. Creates the cleaned dataset and verifies its annotations.

    Raises:
        FileExistsError: If the output directory already exists.
        ValueError: If the source dataset fails verification.
    """
    if output_root.exists():
        raise FileExistsError(
            f"Output dataset already exists: {output_root}"
        )

    verify_dataset(
        original_root,
        FIVE_CLASS_NAMES,
    )

    wrinkle_sources = collect_wrinkle_sources(
        original_root
    )

    source_ids = sorted(wrinkle_sources)

    rng = random.Random(SEED)

    validation_source, test_source = rng.sample(
        source_ids,
        2,
    )

    training_sources = [
        source_id
        for source_id in source_ids
        if source_id not in {
            validation_source,
            test_source,
        }
    ]

    print("\nSOURCE-LEVEL SPLIT")
    print("Training sources:", training_sources)
    print("Validation source:", validation_source)
    print("Test source:", test_source)

    shutil.copytree(
        original_root,
        output_root,
    )

    move_source_group(
        output_root,
        validation_source,
        "valid",
    )

    move_source_group(
        output_root,
        test_source,
        "test",
    )

    write_dataset_yaml(
        output_root,
        FIVE_CLASS_NAMES,
    )

    verify_dataset(
        output_root,
        FIVE_CLASS_NAMES,
    )

    print("\nCleaned five-class dataset created:", output_root)


def remove_wrinkle_images(
    dataset_root: Path,
) -> dict[str, int]:
    """Remove entire image–annotation pairs containing wrinkle defects.

    Any image with at least one class-4 annotation is excluded,
    including images that also contain other defect classes.
    This prevents visible wrinkle defects from being treated
    as background after the class is removed.

    Args:
        dataset_root: Working copy of the cleaned five-class dataset.

    Returns:
        Dictionary giving the number of removed images per split.

    Raises:
        FileNotFoundError: If an annotation selected for removal
            does not have its corresponding image.
    """
    removed = {}

    for split in SPLITS:
        labels_dir = dataset_root / split / "labels"

        removal_pairs = []

        for image_path in list_images(dataset_root, split):
            label_path = labels_dir / f"{image_path.stem}.txt"

            labels, _ = read_yolo_annotations(
                label_path,
                num_classes=len(FIVE_CLASS_NAMES),
                allow_missing=True,
            )

            if WRINKLE_CLASS_ID in labels:
                if not label_path.is_file():
                    raise FileNotFoundError(
                        f"Missing wrinkle label file: {label_path}"
                    )

                removal_pairs.append(
                    (image_path, label_path)
                )

        for image_path, label_path in removal_pairs:
            image_path.unlink()
            label_path.unlink()

        removed[split] = len(removal_pairs)

    return removed


def prepare_four_class(
    clean_root: Path,
    output_root: Path,
) -> None:
    """Create the final four-class dataset from the cleaned dataset.

    The cleaned five-class dataset is copied before any modification.
    Images containing wrinkle annotations are removed together with
    their labels, and the dataset configuration is updated.

    Args:
        clean_root: Existing cleaned five-class dataset.
        output_root: New final four-class output dataset.

    Returns:
        None. Creates and verifies the four-class dataset.

    Raises:
        FileExistsError: If the four-class output directory exists.
        ValueError: If the cleaned dataset has an invalid class mapping.
    """
    if output_root.exists():
        raise FileExistsError(
            f"Output dataset already exists: {output_root}"
        )

    verify_dataset(
        clean_root,
        FIVE_CLASS_NAMES,
    )

    shutil.copytree(
        clean_root,
        output_root,
    )

    removed = remove_wrinkle_images(
        output_root
    )

    print("\nREMOVED WRINKLE-CONTAINING IMAGES")

    for split, count in removed.items():
        print(f"{split}: {count}")

    write_dataset_yaml(
        output_root,
        FOUR_CLASS_NAMES,
    )

    verify_dataset(
        output_root,
        FOUR_CLASS_NAMES,
    )

    print("\nFinal four-class dataset created:", output_root)


def main() -> None:
    """Execute the selected dataset-preparation stages.

    Args:
        None. Input project path and preparation stage are supplied
        through command-line arguments.

    Returns:
        None. Writes new prepared datasets inside the project's
        dataset directory.

    Raises:
        FileNotFoundError: If the required source dataset is missing.
        FileExistsError: If any requested output dataset already exists.
        ValueError: If source annotation or class validation fails.
    """
    args = parse_args()

    project_root = (
        args.project_root.expanduser().resolve()
    )

    dataset_directory = project_root / "dataset"

    original_root = (
        dataset_directory / ORIGINAL_DATASET
    )

    clean_root = (
        dataset_directory / CLEAN_DATASET
    )

    four_class_root = (
        dataset_directory / FOUR_CLASS_DATASET
    )

    print("MMA3001 DATASET PREPARATION")
    print("Stage:", args.stage)
    print("Project:", project_root)
    print("Random seed:", SEED)

    if args.stage in ("clean-five", "all"):
        if clean_root.exists():
            raise FileExistsError(
                "Cleaned five-class dataset already exists. "
                "The original dataset will not be overwritten."
            )

        prepare_five_class(
            original_root,
            clean_root,
        )

    if args.stage in ("make-four", "all"):
        if four_class_root.exists():
            raise FileExistsError(
                "Four-class dataset already exists. "
                "The original dataset will not be overwritten."
            )

        prepare_four_class(
            clean_root,
            four_class_root,
        )

    print("\nDATASET PREPARATION COMPLETED")


if __name__ == "__main__":
    main()
