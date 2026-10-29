"""Ordinal (CORN) output handling in NumPy, so every number in the analysis is testable without a GPU.

CORN (Shi, Cao & Raschka, 2023) predicts K-1 conditional probabilities
    c_k = P(y > k | y > k-1),   k = 0..K-2
whose running product gives the survival function S_k = P(y > k).
"""
from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def corn_survival(logits: np.ndarray) -> np.ndarray:
    """(N, K-1) logits -> (N, K-1) survival probabilities P(y > k)."""
    return np.cumprod(sigmoid(logits), axis=1)


def corn_class_probs(logits: np.ndarray) -> np.ndarray:
    """(N, K-1) logits -> (N, K) class probabilities, rows sum to 1 and are non-negative."""
    s = corn_survival(logits)
    n = s.shape[0]
    upper = np.concatenate([np.ones((n, 1)), s], axis=1)    # P(y > k-1), with P(y > -1) = 1
    lower = np.concatenate([s, np.zeros((n, 1))], axis=1)   # P(y > k),   with P(y > K-1) = 0
    return np.clip(upper - lower, 0.0, 1.0)


def corn_predict(logits: np.ndarray) -> np.ndarray:
    """Median of the predictive distribution: the number of thresholds with P(y > k) > 0.5."""
    return (corn_survival(logits) > 0.5).sum(axis=1)


def outputs_to_probs(logits: np.ndarray, head: str, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    if head == "corn":
        return corn_class_probs(z)
    if head == "ce":
        return softmax(z)
    raise ValueError(f"unknown head {head!r}")


def outputs_to_pred(logits: np.ndarray, head: str) -> np.ndarray:
    """Predicted grade, always from the raw (untempered) logits.

    For softmax, temperature never changes the argmax. For CORN it can: the survival
    function is a product of sigmoids, so rescaling logits can move a product across 0.5.
    Calibration is therefore applied to confidences only, never to predictions.
    """
    if head == "corn":
        return corn_predict(logits)
    return logits.argmax(axis=1)
