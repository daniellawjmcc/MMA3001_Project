
"""Unit tests for shared detection matching and threshold evaluation."""

import numpy as np
import pandas as pd
import pytest
import torch

from mma3001.evaluation import (
    detection_confusion_matrix,
    evaluate_threshold,
    evaluate_thresholds,
    select_best_threshold,
    correct_detections_table,
    per_class_recall_table,
)


@pytest.fixture
def sample_records():
    """Create two-class predictions with known matching outcomes."""
    predictions = [{
        "boxes": torch.tensor([
            [0, 0, 10, 10],
            [20, 20, 30, 30],
            [40, 40, 50, 50],
        ], dtype=torch.float32),
        "scores": torch.tensor([0.90, 0.80, 0.40]),
        "labels": torch.tensor([0, 1, 0]),
    }]

    targets = [{
        "boxes": torch.tensor([
            [0, 0, 10, 10],
            [20, 20, 30, 30],
        ], dtype=torch.float32),
        "labels": torch.tensor([0, 1]),
    }]

    return predictions, targets


def test_confusion_matrix_with_false_positive(sample_records):
    """Check correct matches and a background false positive."""
    predictions, targets = sample_records

    matrix = detection_confusion_matrix(
        predictions, targets,
        confidence_threshold=0.30,
        num_classes=2,
    )

    expected = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [0, 0, 0],
    ])

    np.testing.assert_array_equal(matrix, expected)


def test_confidence_filtering(sample_records):
    """Verify that low-confidence predictions are removed."""
    predictions, targets = sample_records

    matrix = detection_confusion_matrix(
        predictions, targets,
        confidence_threshold=0.85,
        num_classes=2,
    )

    expected = np.array([
        [1, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
    ])

    np.testing.assert_array_equal(matrix, expected)


def test_wrong_class_prediction():
    """An overlapping wrong-class prediction must be off-diagonal."""
    predictions = [{
        "boxes": torch.tensor([[0, 0, 10, 10]], dtype=torch.float32),
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([1]),
    }]

    targets = [{
        "boxes": torch.tensor([[0, 0, 10, 10]], dtype=torch.float32),
        "labels": torch.tensor([0]),
    }]

    matrix = detection_confusion_matrix(
        predictions, targets, 0.5, num_classes=2
    )

    assert matrix[1, 0] == 1
    assert matrix.sum() == 1


def test_duplicate_detection():
    """One ground-truth object cannot match two predictions."""
    predictions = [{
        "boxes": torch.tensor([
            [0, 0, 10, 10],
            [0, 0, 10, 10],
        ], dtype=torch.float32),
        "scores": torch.tensor([0.9, 0.8]),
        "labels": torch.tensor([0, 0]),
    }]

    targets = [{
        "boxes": torch.tensor([[0, 0, 10, 10]], dtype=torch.float32),
        "labels": torch.tensor([0]),
    }]

    matrix = detection_confusion_matrix(
        predictions, targets, 0.5, num_classes=1
    )

    assert matrix[0, 0] == 1
    assert matrix[0, 1] == 1


def test_one_based_class_ids():
    """RetinaNet-style class IDs should map to zero-based matrix indices."""
    predictions = [{
        "boxes": torch.tensor([[0, 0, 10, 10]], dtype=torch.float32),
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([1]),
    }]

    targets = [{
        "boxes": torch.tensor([[0, 0, 10, 10]], dtype=torch.float32),
        "labels": torch.tensor([1]),
    }]

    matrix = detection_confusion_matrix(
        predictions, targets, 0.5,
        num_classes=2,
        label_offset=1,
    )

    assert matrix[0, 0] == 1


def test_evaluate_threshold(sample_records):
    """Overall and per-class metrics must agree with the matrix."""
    predictions, targets = sample_records

    overall, per_class, matrix = evaluate_threshold(
        predictions,
        targets,
        confidence_threshold=0.30,
        class_names=["class-a", "class-b"],
    )

    assert overall["TP"] == 2
    assert overall["FP"] == 1
    assert overall["FN"] == 0
    assert overall["Precision"] == pytest.approx(2 / 3)
    assert overall["Recall"] == pytest.approx(1.0)
    assert len(per_class) == 2
    assert matrix.shape == (3, 3)


def test_threshold_sweep(sample_records):
    """Threshold sweeps should return matching rows and matrices."""
    predictions, targets = sample_records

    overall, classes, matrices = evaluate_thresholds(
        predictions,
        targets,
        thresholds=[0.30, 0.85],
        class_names=["class-a", "class-b"],
    )

    assert len(overall) == 2
    assert len(classes) == 4
    assert len(matrices) == 2
    assert overall.iloc[0]["TP"] == 2
    assert overall.iloc[1]["TP"] == 1


def test_threshold_selection_tie():
    """When scores tie, select the lowest threshold."""
    results = pd.DataFrame({
        "Threshold": [0.5, 0.2, 0.8],
        "F1": [0.7, 0.7, 0.5],
    })

    selected = select_best_threshold(results)

    assert selected["Threshold"] == pytest.approx(0.2)


def test_threshold_tables(sample_records):
    """Correct-detection and recall tables should have class columns."""
    predictions, targets = sample_records

    _, classes, _ = evaluate_thresholds(
        predictions,
        targets,
        thresholds=[0.30, 0.85],
        class_names=["class-a", "class-b"],
    )

    tp_table = correct_detections_table(classes)
    recall_table = per_class_recall_table(classes)

    assert tp_table.loc[0.30, "class-a"] == 1
    assert tp_table.loc[0.85, "class-b"] == 0
    assert recall_table.loc[0.85, "class-b"] == 0.0


def test_invalid_confidence(sample_records):
    """Reject confidence thresholds outside the permitted range."""
    predictions, targets = sample_records

    with pytest.raises(ValueError):
        detection_confusion_matrix(
            predictions, targets, 1.5, num_classes=2
        )
