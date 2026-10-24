
"""Reusable plotting utilities for MMA3001 object detection experiments.

This module displays saved experiment figures and visualises model
training histories, confusion matrices and confidence-threshold metrics.

Metric calculation is handled by metrics.py and evaluation.py.
These functions do not modify datasets, checkpoints or source figures.
"""

from pathlib import Path
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from PIL import Image


def show_saved_figure(
    image_path: str | Path,
    figsize: tuple[float, float] = (10, 7),
    title: str | None = None,
):
    """Display an existing experiment figure without modifying the file.

    Args:
        image_path: Path to the original saved PNG or JPEG image.
        figsize: Display dimensions in inches.
        title: Optional title displayed above the image.

    Returns:
        Matplotlib Figure containing the original image.

    Raises:
        FileNotFoundError: If the image does not exist.
    """
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Saved figure not found: {image_path}"
        )

    # Copy the image into memory without changing the saved original.
    with Image.open(image_path) as source:
        image = source.copy()

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)

    if title:
        ax.set_title(title)

    ax.axis("off")
    fig.tight_layout()

    return fig


def normalize_confusion_matrix(
    matrix: np.ndarray,
) -> np.ndarray:
    """Normalise a detection confusion matrix by true-class totals.

    Confusion matrices use predicted classes in rows and true classes
    in columns. Therefore, each column is normalised independently.

    Args:
        matrix: Square confusion matrix containing detection counts.

    Returns:
        Floating-point confusion matrix with normalised columns.

    Raises:
        ValueError: If the input is not a square matrix or contains
            invalid count values.
    """
    matrix = np.asarray(matrix)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "Confusion matrix must be square."
        )

    if (
        not np.issubdtype(matrix.dtype, np.number)
        or not np.all(np.isfinite(matrix))
        or np.any(matrix < 0)
        or not np.all(matrix == np.floor(matrix))
    ):
        raise ValueError(
            "Confusion matrix must contain non-negative integer counts."
        )

    # Column totals represent the number of true objects in each class.
    totals = matrix.sum(axis=0, keepdims=True)

    normalized = np.divide(
        matrix.astype(float),
        totals,
        out=np.zeros_like(matrix, dtype=float),
        where=totals != 0,
    )

    return normalized


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str],
    title: str = "Detection Confusion Matrix",
    normalized: bool = False,
    ax=None,
    cmap: str = "Blues",
    vmax: float | None = None,
    annotate: bool = True,
):
    """Plot a detection confusion matrix including background.

    Args:
        matrix: Matrix of shape (C + 1, C + 1).
        class_names: Names of the C foreground defect classes.
        title: Figure title.
        normalized: Whether to normalise by true-class totals.
        ax: Optional existing Matplotlib Axes.
        cmap: Matplotlib colour map.
        vmax: Optional maximum colour-scale value.
        annotate: Whether to show nonzero cell values.

    Returns:
        Matplotlib Axes containing the confusion matrix.

    Raises:
        ValueError: If matrix dimensions do not match class names.
    """
    matrix = np.asarray(matrix)
    class_names = list(class_names)

    expected_shape = (
        len(class_names) + 1,
        len(class_names) + 1,
    )

    if matrix.shape != expected_shape:
        raise ValueError(
            f"Expected matrix shape {expected_shape}, "
            f"received {matrix.shape}."
        )

    # Validate counts even when a raw rather than normalised
    # confusion matrix is requested.
    normalize_confusion_matrix(matrix)

    values = (
        normalize_confusion_matrix(matrix)
        if normalized
        else matrix
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 7))

    # Include unmatched detections and missed objects as background.
    labels = class_names + ["background"]

    color_max = (
        float(vmax)
        if vmax is not None
        else (1.0 if normalized else max(float(values.max()), 1.0))
    )

    if color_max <= 0:
        raise ValueError("vmax must be positive.")

    ax.imshow(
        values,
        cmap=cmap,
        vmin=0,
        vmax=color_max,
    )

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))

    ax.set_xticklabels(
        labels,
        rotation=55,
        ha="right",
    )
    ax.set_yticklabels(labels)

    ax.set_xlabel("True Class")
    ax.set_ylabel("Predicted Class")
    ax.set_title(title)

    if annotate:
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = float(values[row, column])

                if value == 0:
                    continue

                text = (
                    f"{value:.2f}"
                    if normalized
                    else str(int(value))
                )

                ax.text(
                    column,
                    row,
                    text,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=(
                        "white"
                        if value > color_max / 2
                        else "black"
                    ),
                )

    return ax


def plot_confusion_matrices(
    matrices: Sequence[np.ndarray],
    thresholds: Sequence[float],
    class_names: Sequence[str],
    normalized: bool = False,
    cmap: str = "Blues",
):
    """Plot threshold-specific confusion matrices side by side.

    All panels share the same colour-scale maximum.

    Args:
        matrices: Sequence of confusion matrices.
        thresholds: Confidence thresholds for the matrices.
        class_names: Foreground class names.
        normalized: Whether to normalise each matrix.
        cmap: Matplotlib colour map.

    Returns:
        Matplotlib Figure containing the matrix comparison.

    Raises:
        ValueError: If the matrix and threshold counts differ.
    """
    matrices = list(matrices)
    thresholds = list(thresholds)

    if not matrices or len(matrices) != len(thresholds):
        raise ValueError(
            "Provide one confusion matrix for each threshold."
        )

    # Use a shared scale for visually comparable threshold panels.
    if normalized:
        maximum = 1.0
    else:
        maximum = max(
            max(float(np.asarray(matrix).max()), 1.0)
            for matrix in matrices
        )

    fig, axes = plt.subplots(
        1,
        len(matrices),
        figsize=(5.5 * len(matrices), 5.5),
        constrained_layout=True,
        squeeze=False,
    )

    for ax, matrix, threshold in zip(
        axes[0], matrices, thresholds
    ):
        plot_confusion_matrix(
            matrix=matrix,
            class_names=class_names,
            title=f"Confidence = {float(threshold):.3f}",
            normalized=normalized,
            ax=ax,
            cmap=cmap,
            vmax=maximum,
        )

        ax.tick_params(labelsize=8)

    figure_type = (
        "Normalised" if normalized else "Raw"
    )

    fig.suptitle(
        f"{figure_type} Confusion Matrices "
        "Across Confidence Thresholds"
    )

    return fig


def plot_training_history(
    history: pd.DataFrame,
    metric_columns: dict[str, str],
    epoch_column: str = "epoch",
    title: str = "Training History",
    ylabel: str = "Metric Value",
):
    """Plot selected metrics from a saved training-history table.

    Args:
        history: DataFrame containing recorded training epochs.
        metric_columns: Mapping from actual CSV column names to
            desired legend labels.
        epoch_column: Column containing epoch numbers.
        title: Figure title.
        ylabel: Vertical-axis label.

    Returns:
        Matplotlib Figure containing the training curves.

    Raises:
        ValueError: If required columns are missing or values invalid.
    """
    required = [epoch_column, *metric_columns]

    missing = [
        column for column in required
        if column not in history.columns
    ]

    if missing:
        raise ValueError(
            f"Training history is missing columns: {missing}"
        )

    if not metric_columns or history.empty:
        raise ValueError(
            "Training history and metric columns cannot be empty."
        )

    # Convert recorded data to numeric values before plotting.
    numeric = history[required].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(
            "Training history contains non-numeric or missing values."
        )

    fig, ax = plt.subplots(figsize=(9, 5))

    for column, label in metric_columns.items():
        ax.plot(
            numeric[epoch_column],
            numeric[column],
            marker="o",
            label=label,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    return fig


def plot_threshold_metrics(
    results: pd.DataFrame,
    metric_columns: Sequence[str] = (
        "Precision",
        "Recall",
        "F1",
    ),
    title: str = "Confidence-Threshold Performance",
    selected_thresholds: dict[str, float] | None = None,
):
    """Plot precision, recall and F1 against confidence threshold.

    Args:
        results: Threshold evaluation DataFrame.
        metric_columns: Existing metric columns to plot.
        title: Figure title.
        selected_thresholds: Optional mapping from description
            to selected threshold, shown as vertical reference lines.

    Returns:
        Matplotlib Figure containing the threshold curves.

    Raises:
        ValueError: If required metrics are missing or invalid.
    """
    required = ["Threshold", *metric_columns]

    missing = [
        column for column in required
        if column not in results.columns
    ]

    if missing or not metric_columns:
        raise ValueError(
            f"Missing or empty metric selection: {missing}"
        )

    numeric = results[required].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if numeric.empty or not np.isfinite(
        numeric.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Threshold data must contain valid numeric values."
        )

    # Sorting avoids unexpected zigzags from unsorted result tables.
    numeric = numeric.sort_values("Threshold")

    fig, ax = plt.subplots(figsize=(10, 6))

    for column in metric_columns:
        ax.plot(
            numeric["Threshold"],
            numeric[column],
            marker="o",
            label=column,
        )

    if selected_thresholds:
        for label, threshold in selected_thresholds.items():
            ax.axvline(
                float(threshold),
                linestyle="--",
                label=label,
            )

    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    return fig


def plot_per_class_recall(
    recall_table: pd.DataFrame,
    title: str = "Per-Class Recall vs Confidence Threshold",
):
    """Plot recall sensitivity for each defect category.

    Args:
        recall_table: Pivot table whose index contains confidence
            thresholds and whose columns contain class names.
        title: Figure title.

    Returns:
        Matplotlib Figure containing per-class recall curves.

    Raises:
        ValueError: If the table is empty or contains invalid data.
    """
    if recall_table.empty:
        raise ValueError("Per-class recall table cannot be empty.")

    # Retain the class columns but sort threshold values.
    table = recall_table.sort_index()

    thresholds = pd.to_numeric(
        pd.Index(table.index),
        errors="coerce",
    )

    values = table.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if (
        not np.isfinite(np.asarray(thresholds, dtype=float)).all()
        or not np.isfinite(values.to_numpy(dtype=float)).all()
    ):
        raise ValueError(
            "Recall table contains missing or non-numeric values."
        )

    fig, ax = plt.subplots(figsize=(10, 6))

    for class_name in values.columns:
        ax.plot(
            thresholds,
            values[class_name],
            marker="o",
            label=str(class_name),
        )

    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Recall")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    return fig
