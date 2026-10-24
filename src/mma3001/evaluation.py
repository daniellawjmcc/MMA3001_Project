
"""Shared object detection evaluation for MMA3001 experiments.

This module evaluates existing model predictions using confidence
thresholds and intersection-over-union (IoU) matching.

Confusion matrices have predicted classes in rows, true classes in
columns, and background in the final row and column.

Model inference, COCO-style average precision and plotting are
handled separately.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from torchvision.ops import box_iou

from .metrics import (
    summarise_confusion_matrix,
)


def validate_detection_records(
    predictions: Sequence[Mapping],
    targets: Sequence[Mapping],
    num_classes: int,
    label_offset: int = 0,
) -> None:
    """Validate model predictions and ground-truth target records.

    Args:
        predictions: Sequence of dictionaries containing boxes,
            scores and labels.
        targets: Sequence of dictionaries containing boxes and labels.
        num_classes: Number of foreground defect categories.
        label_offset: Smallest valid class ID, usually zero or one.

    Raises:
        ValueError: If detection arrays or class IDs are invalid.
    """
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer.")

    if label_offset not in (0, 1):
        raise ValueError("label_offset must be zero or one.")

    if len(predictions) != len(targets):
        raise ValueError(
            "The numbers of prediction and target records must match."
        )

    for image_index, (prediction, target) in enumerate(
        zip(predictions, targets)
    ):
        for name, record in [
            ("Prediction", prediction),
            ("Target", target),
        ]:
            if "boxes" not in record or "labels" not in record:
                raise ValueError(
                    f"{name} record {image_index} needs boxes and labels."
                )

            boxes = torch.as_tensor(record["boxes"])
            labels = torch.as_tensor(record["labels"])

            if boxes.ndim != 2 or boxes.shape[1] != 4:
                raise ValueError(
                    f"{name} boxes must have shape (N, 4)."
                )

            if labels.ndim != 1 or len(labels) != len(boxes):
                raise ValueError(
                    f"{name} labels must match its bounding boxes."
                )

            if not torch.isfinite(boxes).all():
                raise ValueError(
                    f"{name} record contains non-finite boxes."
                )

            # All boxes must have positive width and height.
            if torch.any(boxes[:, 2:] <= boxes[:, :2]):
                raise ValueError(
                    f"{name} record contains zero-area boxes."
                )

            if labels.dtype == torch.bool or labels.is_floating_point():
                raise ValueError(
                    f"{name} class IDs must be integer values."
                )

            if torch.any(labels < label_offset) or torch.any(
                labels >= label_offset + num_classes
            ):
                raise ValueError(
                    f"{name} record contains invalid class IDs."
                )

        scores = torch.as_tensor(prediction.get("scores", []))

        if scores.ndim != 1 or len(scores) != len(
            prediction["boxes"]
        ):
            raise ValueError(
                f"Prediction scores must match boxes in image {image_index}."
            )

        if not torch.isfinite(scores).all() or torch.any(
            (scores < 0) | (scores > 1)
        ):
            raise ValueError(
                f"Prediction image {image_index} has invalid scores."
            )


def detection_confusion_matrix(
    predictions: Sequence[Mapping],
    targets: Sequence[Mapping],
    confidence_threshold: float,
    num_classes: int,
    iou_threshold: float = 0.50,
    label_offset: int = 0,
    validate: bool = True,
) -> np.ndarray:
    """Generate a detection confusion matrix with background.

    Predictions are filtered by confidence and matched in descending
    score order to unused ground-truth boxes using maximum IoU.

    Matches are determined by IoU independently of class label.
    Therefore, incorrect-class matches appear off the matrix diagonal.

    Args:
        predictions: Detection dictionaries with boxes, scores, labels.
        targets: Ground-truth dictionaries with boxes and labels.
        confidence_threshold: Minimum prediction confidence.
        num_classes: Number of defect categories.
        iou_threshold: Minimum IoU for a matched detection.
        label_offset: Zero for YOLO; one for the original Torchvision
            label convention.
        validate: Whether to validate all input records.

    Returns:
        Square integer matrix with shape (C + 1, C + 1).

    Raises:
        ValueError: If thresholds or input records are invalid.
    """
    if not 0 <= confidence_threshold <= 1:
        raise ValueError(
            "Confidence threshold must be between zero and one."
        )

    if not 0 < iou_threshold <= 1:
        raise ValueError(
            "IoU threshold must be greater than zero and at most one."
        )

    if validate:
        validate_detection_records(
            predictions,
            targets,
            num_classes,
            label_offset,
        )

    background = num_classes

    matrix = np.zeros(
        (num_classes + 1, num_classes + 1),
        dtype=np.int64,
    )

    for prediction, target in zip(predictions, targets):
        pred_boxes = torch.as_tensor(
            prediction["boxes"], dtype=torch.float32
        ).cpu()

        pred_scores = torch.as_tensor(
            prediction["scores"], dtype=torch.float32
        ).cpu()

        pred_labels = torch.as_tensor(
            prediction["labels"], dtype=torch.int64
        ).cpu()

        gt_boxes = torch.as_tensor(
            target["boxes"], dtype=torch.float32
        ).cpu()

        gt_labels = torch.as_tensor(
            target["labels"], dtype=torch.int64
        ).cpu()

        # Filter predictions without changing the original arrays.
        keep = pred_scores >= confidence_threshold

        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]

        # Higher-confidence predictions receive matching priority.
        order = torch.argsort(
            pred_scores, descending=True, stable=True
        )

        pred_boxes = pred_boxes[order]
        pred_labels = pred_labels[order]

        matched_gt = set()

        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            predicted_class = int(pred_label) - label_offset

            if len(gt_boxes) == 0:
                matrix[predicted_class, background] += 1
                continue

            ious = box_iou(
                pred_box.unsqueeze(0),
                gt_boxes,
            )[0]

            # Exclude ground-truth objects that are already matched.
            for matched_index in matched_gt:
                ious[matched_index] = -1.0

            best_iou, best_index = ious.max(dim=0)
            best_index = int(best_index)

            if float(best_iou) >= iou_threshold:
                true_class = int(
                    gt_labels[best_index]
                ) - label_offset

                matrix[predicted_class, true_class] += 1
                matched_gt.add(best_index)
            else:
                # Predictions with no IoU match are false positives.
                matrix[predicted_class, background] += 1

        # Unmatched ground-truth boxes are false negatives.
        for gt_index, gt_label in enumerate(gt_labels):
            if gt_index not in matched_gt:
                true_class = int(gt_label) - label_offset
                matrix[background, true_class] += 1

    return matrix


def evaluate_threshold(
    predictions: Sequence[Mapping],
    targets: Sequence[Mapping],
    confidence_threshold: float,
    class_names: Sequence[str],
    iou_threshold: float = 0.50,
    label_offset: int = 0,
    validate: bool = True,
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    """Evaluate overall and per-class metrics at one confidence threshold.

    Args:
        predictions: Existing model predictions.
        targets: Matching ground-truth annotations.
        confidence_threshold: Minimum detection confidence.
        class_names: Class names in dataset class-ID order.
        iou_threshold: Detection matching IoU threshold.
        label_offset: Zero-based or one-based label convention.
        validate: Whether to validate input records.

    Returns:
        Tuple containing:
            - Overall TP, FP, FN, micro and macro metrics.
            - Per-class detection metrics DataFrame.
            - Confusion matrix including background.
    """
    class_names = list(class_names)

    matrix = detection_confusion_matrix(
        predictions=predictions,
        targets=targets,
        confidence_threshold=confidence_threshold,
        num_classes=len(class_names),
        iou_threshold=iou_threshold,
        label_offset=label_offset,
        validate=validate,
    )

    overall, per_class = summarise_confusion_matrix(
        matrix,
        class_names,
    )

    overall = {
        "Threshold": float(confidence_threshold),
        **overall,
    }

    per_class = per_class.copy()
    per_class.insert(0, "Threshold", float(confidence_threshold))

    return overall, per_class, matrix


def evaluate_thresholds(
    predictions: Sequence[Mapping],
    targets: Sequence[Mapping],
    thresholds: Sequence[float],
    class_names: Sequence[str],
    iou_threshold: float = 0.50,
    label_offset: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, list[np.ndarray]]:
    """Evaluate several confidence thresholds using one matching rule.

    Args:
        predictions: Existing model predictions.
        targets: Matching ground-truth annotations.
        thresholds: Confidence threshold values to evaluate.
        class_names: Foreground class names in class-ID order.
        iou_threshold: Minimum detection matching IoU.
        label_offset: Original model class-label offset.

    Returns:
        Tuple containing:
            - Overall threshold-comparison DataFrame.
            - Per-class threshold-comparison DataFrame.
            - Confusion matrices in the order of the thresholds.

    Raises:
        ValueError: If no thresholds are supplied.
    """
    thresholds = [float(value) for value in thresholds]

    if not thresholds:
        raise ValueError("At least one threshold is required.")

    # Validate once before the repeated confidence sweep.
    validate_detection_records(
        predictions,
        targets,
        num_classes=len(class_names),
        label_offset=label_offset,
    )

    overall_rows = []
    per_class_frames = []
    matrices = []

    for threshold in thresholds:
        overall, per_class, matrix = evaluate_threshold(
            predictions=predictions,
            targets=targets,
            confidence_threshold=threshold,
            class_names=class_names,
            iou_threshold=iou_threshold,
            label_offset=label_offset,
            validate=False,
        )

        overall_rows.append(overall)
        per_class_frames.append(per_class)
        matrices.append(matrix)

    overall_df = pd.DataFrame(overall_rows)

    per_class_df = pd.concat(
        per_class_frames,
        ignore_index=True,
    )

    return overall_df, per_class_df, matrices


def select_best_threshold(
    threshold_results: pd.DataFrame,
    metric: str = "F1",
) -> pd.Series:
    """Select the best validation confidence threshold.

    Ties are resolved by choosing the lowest confidence threshold,
    making the selected operating point deterministic.

    Args:
        threshold_results: DataFrame generated by evaluate_thresholds.
        metric: Metric column to maximise, such as F1 or Macro-F1.

    Returns:
        Series representing the selected threshold and its metrics.

    Raises:
        ValueError: If the required metric or valid rows are missing.
    """
    required_columns = {"Threshold", metric}

    if not required_columns.issubset(threshold_results.columns):
        raise ValueError(
            f"Threshold results must contain {required_columns}."
        )

    valid_rows = threshold_results.dropna(
        subset=["Threshold", metric]
    )

    if valid_rows.empty:
        raise ValueError(
            "No valid threshold metrics are available."
        )

    # Use a stable tie-breaking rule for reproducible selection.
    ranked = valid_rows.sort_values(
        by=[metric, "Threshold"],
        ascending=[False, True],
        kind="stable",
    )

    return ranked.iloc[0].copy()


def correct_detections_table(
    per_class_results: pd.DataFrame,
) -> pd.DataFrame:
    """Create a correct-detections-per-class threshold table.

    Args:
        per_class_results: DataFrame returned by evaluate_thresholds.

    Returns:
        Pivot table containing true-positive counts for each class.
    """
    return per_class_results.pivot(
        index="Threshold",
        columns="Class",
        values="TP",
    )


def per_class_recall_table(
    per_class_results: pd.DataFrame,
) -> pd.DataFrame:
    """Create per-class recall values across confidence thresholds.

    Args:
        per_class_results: DataFrame returned by evaluate_thresholds.

    Returns:
        Pivot table containing recall values by class and threshold.
    """
    return per_class_results.pivot(
        index="Threshold",
        columns="Class",
        values="Recall",
    )
