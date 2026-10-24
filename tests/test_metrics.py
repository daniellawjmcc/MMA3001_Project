
"""Unit tests for shared precision, recall and F1 calculations."""

import numpy as np
import pandas as pd
import pytest

from mma3001.metrics import (
    precision_recall_f1,
    counts_from_confusion_matrix,
    per_class_metrics,
    overall_detection_metrics,
    macro_average_metrics,
    summarise_confusion_matrix,
    add_average_precision,
)


def test_precision_recall_f1():
    """Verify detection scores using known TP, FP and FN counts."""
    result = precision_recall_f1(tp=8, fp=2, fn=2)

    assert result["Precision"] == pytest.approx(0.8)
    assert result["Recall"] == pytest.approx(0.8)
    assert result["F1"] == pytest.approx(0.8)


def test_zero_predictions():
    """Undefined metrics should follow the project's zero convention."""
    result = precision_recall_f1(tp=0, fp=0, fn=5)

    assert result["Precision"] == 0.0
    assert result["Recall"] == 0.0
    assert result["F1"] == 0.0


def test_invalid_counts():
    """Negative and non-integer detection counts must be rejected."""
    with pytest.raises(ValueError):
        precision_recall_f1(tp=-1, fp=0, fn=0)

    with pytest.raises(ValueError):
        precision_recall_f1(tp=1.5, fp=0, fn=0)


@pytest.fixture
def sample_matrix():
    """Return a two-class detection matrix including background."""
    return np.array([
        [8, 1, 2],
        [1, 6, 1],
        [2, 3, 0],
    ])


def test_counts_from_confusion_matrix(sample_matrix):
    """Verify class-confusion and background counts."""
    result = counts_from_confusion_matrix(
        sample_matrix,
        num_classes=2,
    )

    assert result.loc[0, "TP"] == 8
    assert result.loc[0, "FP"] == 3
    assert result.loc[0, "FN"] == 3

    assert result.loc[1, "TP"] == 6
    assert result.loc[1, "FP"] == 2
    assert result.loc[1, "FN"] == 4


def test_overall_detection_metrics(sample_matrix):
    """Check the micro-averaged TP, FP and FN totals."""
    result = overall_detection_metrics(
        sample_matrix,
        num_classes=2,
    )

    assert result["TP"] == 14
    assert result["FP"] == 5
    assert result["FN"] == 7

    assert result["Precision"] == pytest.approx(14 / 19)
    assert result["Recall"] == pytest.approx(14 / 21)


def test_per_class_metrics(sample_matrix):
    """Verify class names and per-class precision and recall."""
    result = per_class_metrics(
        sample_matrix,
        class_names=["class-a", "class-b"],
    )

    assert result.loc[0, "Class"] == "class-a"
    assert result.loc[1, "Class"] == "class-b"

    assert result.loc[0, "Precision"] == pytest.approx(8 / 11)
    assert result.loc[1, "Recall"] == pytest.approx(6 / 10)


def test_macro_average_metrics():
    """Ensure each class contributes equally to macro averages."""
    values = pd.DataFrame({
        "Precision": [0.8, 0.4],
        "Recall": [0.6, 0.2],
        "F1": [0.7, 0.3],
    })

    result = macro_average_metrics(values)

    assert result["Macro-Precision"] == pytest.approx(0.6)
    assert result["Macro-Recall"] == pytest.approx(0.4)
    assert result["Macro-F1"] == pytest.approx(0.5)


def test_summarise_confusion_matrix(sample_matrix):
    """Check combined overall and per-class metric reporting."""
    overall, classes = summarise_confusion_matrix(
        sample_matrix,
        class_names=["class-a", "class-b"],
    )

    assert overall["TP"] == 14
    assert len(classes) == 2
    assert "Macro-F1" in overall


def test_add_average_precision():
    """Ensure supplied AP values join by class name."""
    class_metrics = pd.DataFrame({
        "Class": ["class-a", "class-b"],
        "TP": [8, 6],
    })

    result = add_average_precision(
        class_metrics,
        {
            "class-a": 0.65,
            "class-b": 0.35,
        },
    )

    assert result.loc[0, "AP@50:95"] == pytest.approx(0.65)
    assert result.loc[1, "AP@50:95"] == pytest.approx(0.35)
