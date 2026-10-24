
"""Unit tests for the shared plotting utilities."""

from pathlib import Path

import matplotlib

# Use a non-interactive backend for automated tests.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from PIL import Image

from mma3001.plotting import (
    show_saved_figure,
    normalize_confusion_matrix,
    plot_confusion_matrix,
    plot_confusion_matrices,
    plot_training_history,
    plot_threshold_metrics,
    plot_per_class_recall,
)


@pytest.fixture
def sample_matrix():
    """Provide a two-class matrix with a background category."""
    return np.array([
        [8, 1, 2],
        [1, 6, 1],
        [2, 3, 0],
    ])


def test_show_saved_figure(tmp_path: Path):
    """Verify that an existing image can be displayed unchanged."""
    image_path = tmp_path / "original.png"

    Image.new("RGB", (100, 60)).save(image_path)
    original_bytes = image_path.read_bytes()

    fig = show_saved_figure(image_path)

    assert fig is not None
    assert image_path.read_bytes() == original_bytes

    plt.close(fig)


def test_missing_saved_figure(tmp_path: Path):
    """Missing figures must produce an explicit error."""
    with pytest.raises(FileNotFoundError):
        show_saved_figure(tmp_path / "missing.png")


def test_normalize_confusion_matrix(sample_matrix):
    """Verify column-wise normalisation by true-class totals."""
    normalized = normalize_confusion_matrix(sample_matrix)

    np.testing.assert_allclose(
        normalized.sum(axis=0),
        [1.0, 1.0, 1.0],
    )

    assert normalized[0, 0] == pytest.approx(8 / 11)
    assert normalized[1, 1] == pytest.approx(6 / 10)


def test_zero_column_normalization():
    """Empty true-class columns must not produce division errors."""
    matrix = np.array([
        [2, 0],
        [1, 0],
    ])

    normalized = normalize_confusion_matrix(matrix)

    assert np.all(np.isfinite(normalized))
    assert np.all(normalized[:, 1] == 0)


def test_plot_confusion_matrix(sample_matrix):
    """Verify that one detection confusion matrix can be plotted."""
    fig, ax = plt.subplots()

    result = plot_confusion_matrix(
        sample_matrix,
        class_names=["class-a", "class-b"],
        ax=ax,
    )

    assert result is ax
    assert len(ax.get_xticklabels()) == 3

    plt.close(fig)


def test_plot_normalized_confusion_matrix(sample_matrix):
    """Verify the normalised confusion-matrix option."""
    fig, ax = plt.subplots()

    plot_confusion_matrix(
        sample_matrix,
        class_names=["class-a", "class-b"],
        normalized=True,
        ax=ax,
    )

    plotted_values = np.asarray(ax.images[0].get_array())

    assert plotted_values[0, 0] == pytest.approx(8 / 11)

    plt.close(fig)


def test_plot_multiple_confusion_matrices(sample_matrix):
    """Verify one panel is produced per confidence threshold."""
    fig = plot_confusion_matrices(
        matrices=[sample_matrix, sample_matrix],
        thresholds=[0.25, 0.50],
        class_names=["class-a", "class-b"],
    )

    assert len(fig.axes) == 2

    plt.close(fig)


def test_plot_training_history():
    """Verify plotting from recorded epoch metrics."""
    history = pd.DataFrame({
        "epoch": [1, 2, 3],
        "train_loss": [1.2, 0.9, 0.6],
        "val_map": [0.10, 0.20, 0.30],
    })

    fig = plot_training_history(
        history,
        metric_columns={
            "train_loss": "Training Loss",
        },
        ylabel="Loss",
    )

    assert len(fig.axes[0].lines) == 1

    plt.close(fig)


def test_plot_threshold_metrics():
    """Verify precision, recall and F1 sensitivity plotting."""
    results = pd.DataFrame({
        "Threshold": [0.1, 0.3, 0.5],
        "Precision": [0.4, 0.6, 0.8],
        "Recall": [0.9, 0.7, 0.5],
        "F1": [0.55, 0.65, 0.62],
    })

    fig = plot_threshold_metrics(
        results,
        selected_thresholds={"Selected F1": 0.3},
    )

    # Three metric lines plus one selected-threshold marker.
    assert len(fig.axes[0].lines) == 4

    plt.close(fig)


def test_plot_per_class_recall():
    """Verify recall curves for multiple defect categories."""
    recall_table = pd.DataFrame({
        "class-a": [0.9, 0.7, 0.5],
        "class-b": [0.8, 0.6, 0.3],
    }, index=[0.1, 0.3, 0.5])

    fig = plot_per_class_recall(recall_table)

    assert len(fig.axes[0].lines) == 2

    plt.close(fig)


def test_invalid_matrix_shape():
    """Reject matrices inconsistent with the supplied classes."""
    matrix = np.zeros((2, 2), dtype=int)

    with pytest.raises(ValueError):
        plot_confusion_matrix(
            matrix,
            class_names=["class-a", "class-b"],
        )


def test_missing_training_column():
    """Do not silently plot unavailable historical metrics."""
    history = pd.DataFrame({
        "epoch": [1, 2],
        "train_loss": [0.9, 0.8],
    })

    with pytest.raises(ValueError):
        plot_training_history(
            history,
            metric_columns={"val_map": "Validation mAP"},
        )
