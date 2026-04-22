"""Unit tests for src/eval.py."""
import numpy as np
import torch

from src.eval import per_class_mean_accuracy, top1_accuracy


def test_top1_accuracy_perfect():
    preds = torch.tensor([0, 1, 2, 3])
    labels = torch.tensor([0, 1, 2, 3])
    assert top1_accuracy(preds, labels) == 1.0


def test_top1_accuracy_half():
    preds = torch.tensor([0, 0, 2, 2])
    labels = torch.tensor([0, 1, 2, 3])
    assert top1_accuracy(preds, labels) == 0.5


def test_per_class_mean_accuracy_equal_weights():
    """If all classes have equal sample counts, per-class mean == top-1."""
    preds = torch.tensor([0, 1, 2, 3])
    labels = torch.tensor([0, 1, 2, 3])
    assert per_class_mean_accuracy(preds, labels, num_classes=4) == 1.0


def test_per_class_mean_accuracy_imbalance():
    """Per-class mean is robust to class imbalance.

    Class 0: 100 samples, 0 correct -> 0%
    Class 1: 1 sample, 1 correct -> 100%
    Top-1 sample-avg = 1/101 ≈ 1%, per-class mean = (0 + 1) / 2 = 50%.
    """
    preds = torch.cat([torch.ones(100, dtype=torch.long),  # wrong for class 0
                       torch.ones(1, dtype=torch.long)])   # right for class 1
    labels = torch.cat([torch.zeros(100, dtype=torch.long),
                        torch.ones(1, dtype=torch.long)])
    top1 = top1_accuracy(preds, labels)
    per_class = per_class_mean_accuracy(preds, labels, num_classes=2)
    assert abs(top1 - 1 / 101) < 1e-6
    assert abs(per_class - 0.5) < 1e-6
