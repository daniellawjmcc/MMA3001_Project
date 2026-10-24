
"""Classification and detection metrics for MMA3001 experiments.

This module calculates precision, recall, F1-score and related
statistics from object detection true-positive, false-positive
and false-negative counts.

Detection matching and confusion-matrix construction are handled
separately in evaluation.py.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def precision_recall_f1(
    tp: int,
    fp: int,
    fn: int,
) -> dict[str, float]:
    """Calculate precision, recall and F1-score from detection counts.

    Undefined precision, recall and F1 values are reported as zero,
    following the convention used in the original project notebooks.

    Args:
        tp: Number of true-positive detections.
        fp: Number of false-positive detections.
        fn: Number of false-negative detections.

    Returns:
        Dictionary containing precision, recall and F1-score.

    Raises:
        ValueError: If any count is negative or is not an integer.
    """
    counts = {"TP": tp, "FP": fp, "FN": fn}

    # Detection counts must be non-negative whole numbers.
    for name, value in counts.items():
        if (
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            or value < 0
        ):
            raise ValueError(
                f"{name} must be a non-negative integer."
            )

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
    }


def counts_from_confusion_matrix(
    matrix: np.ndarray,
    num_classes: int,
) -> pd.DataFrame:
    """Extract per-class TP, FP and FN from a detection confusion matrix.

    The matrix must use predicted classes as rows and true classes as
    columns. The final row and column represent background.

    Args:
        matrix: Square confusion matrix with shape (C + 1, C + 1).
        num_classes: Number of foreground defect classes.

    Returns:
        DataFrame containing Class ID, TP, FP and FN.

    Raises:
        ValueError: If the matrix dimensions or counts are invalid.
    """
    matrix = np.asarray(matrix)

    # Validate the foreground and background matrix dimensions.
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer.")

    expected_shape = (num_classes + 1, num_classes + 1)

    if matrix.shape != expected_shape:
        raise ValueError(
            f"Expected matrix shape {expected_shape}, "
            f"received {matrix.shape}."
        )

    # A confusion matrix must contain non-negative integer counts.
    if (
        not np.issubdtype(matrix.dtype, np.number)
        or not np.all(np.isfinite(matrix))
        or np.any(matrix < 0)
        or not np.all(matrix == np.floor(matrix))
    ):
        raise ValueError(
            "Confusion matrix must contain non-negative integer counts."
        )

    rows = []

    for class_id in range(num_classes):
        # Diagonal counts are correct detections for this class.
        tp = int(matrix[class_id, class_id])

        # The entire predicted-class row includes false positives.
        fp = int(matrix[class_id, :].sum() - tp)

        # The entire true-class column includes missed detections.
        fn = int(matrix[:, class_id].sum() - tp)

        rows.append({
            "Class ID": class_id,
            "TP": tp,
            "FP": fp,
            "FN": fn,
        })

    return pd.DataFrame(rows)


def per_class_metrics(
    matrix: np.ndarray,
    class_names: Sequence[str],
) -> pd.DataFrame:
    """Calculate detection metrics for every foreground class.

    Args:
        matrix: Detection confusion matrix with background in its
            final row and column.
        class_names: Ordered names of the foreground defect classes.

    Returns:
        DataFrame containing class names, TP, FP, FN, precision,
        recall and F1-score.
    """
    # Use the class order supplied by the calling experiment.
    class_names = list(class_names)

    if not class_names:
        raise ValueError("At least one class name is required.")

    counts_df = counts_from_confusion_matrix(
        matrix,
        num_classes=len(class_names),
    )

    rows = []

    for _, count in counts_df.iterrows():
        class_id = int(count["Class ID"])
        tp = int(count["TP"])
        fp = int(count["FP"])
        fn = int(count["FN"])

        scores = precision_recall_f1(tp, fp, fn)

        rows.append({
            "Class": class_names[class_id],
            "TP": tp,
            "FP": fp,
            "FN": fn,
            **scores,
        })

    return pd.DataFrame(rows)


def overall_detection_metrics(
    matrix: np.ndarray,
    num_classes: int,
) -> dict[str, float | int]:
    """Calculate overall micro-averaged detection metrics.

    The total true positives equal the foreground diagonal sum.
    False positives and false negatives include class-confusion
    errors and unmatched background detections.

    Args:
        matrix: Detection confusion matrix with background.
        num_classes: Number of foreground defect classes.

    Returns:
        Dictionary containing TP, FP, FN, precision, recall and F1.
    """
    counts_df = counts_from_confusion_matrix(
        matrix,
        num_classes,
    )

    tp = int(counts_df["TP"].sum())
    fp = int(counts_df["FP"].sum())
    fn = int(counts_df["FN"].sum())

    scores = precision_recall_f1(tp, fp, fn)

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        **scores,
    }


def macro_average_metrics(
    class_metrics: pd.DataFrame,
) -> dict[str, float]:
    """Calculate unweighted averages of per-class detection scores.

    Each class contributes equally, regardless of annotation frequency.
    Missing classes should be treated consistently across experiments.

    Args:
        class_metrics: DataFrame containing Precision, Recall and F1
            columns, with one row per foreground class.

    Returns:
        Dictionary with Macro-Precision, Macro-Recall and Macro-F1.

    Raises:
        ValueError: If the required columns are missing or empty.
    """
    required = ["Precision", "Recall", "F1"]

    if class_metrics.empty or any(
        column not in class_metrics.columns
        for column in required
    ):
        raise ValueError(
            "Class metrics must contain Precision, Recall and F1."
        )

    values = class_metrics[required].to_numpy(dtype=float)

    if not np.all(np.isfinite(values)):
        raise ValueError("Class metrics contain non-finite values.")

    return {
        "Macro-Precision": float(values[:, 0].mean()),
        "Macro-Recall": float(values[:, 1].mean()),
        "Macro-F1": float(values[:, 2].mean()),
    }


def summarise_confusion_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str],
) -> tuple[dict, pd.DataFrame]:
    """Summarise overall and class-specific detection performance.

    Args:
        matrix: Detection confusion matrix including background.
        class_names: Foreground class names in class-ID order.

    Returns:
        Tuple containing an overall metrics dictionary and
        a DataFrame of per-class detection metrics.
    """
    class_names = list(class_names)

    class_df = per_class_metrics(
        matrix,
        class_names=class_names,
    )

    overall = overall_detection_metrics(
        matrix,
        num_classes=len(class_names),
    )

    macro = macro_average_metrics(class_df)

    return {**overall, **macro}, class_df


def add_average_precision(
    class_metrics: pd.DataFrame,
    ap_values: Mapping[str, float],
    column_name: str = "AP@50:95",
) -> pd.DataFrame:
    """Combine detection-count metrics with separately calculated AP.

    Average precision is derived from confidence-ranked detections
    across IoU thresholds, not from one confusion matrix.

    This function only joins existing AP values; it never attempts
    to calculate AP from a fixed-confidence confusion matrix.

    Args:
        class_metrics: DataFrame containing a Class column.
        ap_values: Mapping from class names to previously calculated AP.
        column_name: Name of the resulting AP column.

    Returns:
        New DataFrame containing the supplied average precision.

    Raises:
        ValueError: If class names are missing or AP is outside [0, 1].
    """
    if "Class" not in class_metrics.columns:
        raise ValueError("Class metrics must contain a Class column.")

    result = class_metrics.copy()

    # Missing AP remains NaN rather than being silently set to zero.
    result[column_name] = result["Class"].map(ap_values)

    provided_values = np.asarray(
        list(ap_values.values()),
        dtype=float,
    )

    if not np.all(np.isfinite(provided_values)) or np.any(
        (provided_values < 0) | (provided_values > 1)
    ):
        raise ValueError(
            "Average precision values must be finite and between 0 and 1."
        )

    return result
