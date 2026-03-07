"""
Loss / objective functions and their gradients.
Both receive *logits* (raw pre-softmax values) from the network.
Handles 1D (single sample) and 2D (batch) logits gracefully.
"""
import numpy as np
from .activations import softmax


def _to_2d(logits):
    """Ensure logits is 2D (batch, classes). Returns (logits_2d, was_1d)."""
    logits = np.asarray(logits, dtype=float)
    if logits.ndim == 1:
        return logits[np.newaxis, :], True
    return logits, False


def _to_int_labels(y_true, batch_size):
    """Convert y_true to 1D integer array of length batch_size."""
    y = np.asarray(y_true)
    if y.ndim == 0:
        y = y.reshape(1)
    if y.ndim == 2:
        # Could be one-hot or column vector
        if y.shape[1] == 1:
            y = y.ravel()
        else:
            y = np.argmax(y, axis=1)
    return y.astype(int)


# ── Cross-Entropy ──────────────────────────────────────────────────────────

def cross_entropy_loss(logits, y_true):
    logits_2d, _ = _to_2d(logits)
    batch_size = logits_2d.shape[0]
    y = _to_int_labels(y_true, batch_size)
    probs = softmax(logits_2d)
    log_probs = np.log(np.clip(probs[np.arange(batch_size), y], 1e-12, 1.0))
    return -np.mean(log_probs)


def cross_entropy_gradient(logits, y_true):
    logits_2d, was_1d = _to_2d(logits)
    batch_size = logits_2d.shape[0]
    y = _to_int_labels(y_true, batch_size)
    probs = softmax(logits_2d)
    dLogits = probs.copy()
    dLogits[np.arange(batch_size), y] -= 1.0
    dLogits /= batch_size
    return dLogits[0] if was_1d else dLogits


# ── MSE ───────────────────────────────────────────────────────────────────

def mse_loss(logits, y_true):
    logits_2d, _ = _to_2d(logits)
    batch_size, num_classes = logits_2d.shape
    y = _to_int_labels(y_true, batch_size)
    probs = softmax(logits_2d)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(batch_size), y] = 1.0
    return np.mean(np.sum((probs - one_hot) ** 2, axis=1))


def mse_gradient(logits, y_true):
    logits_2d, was_1d = _to_2d(logits)
    batch_size, num_classes = logits_2d.shape
    y = _to_int_labels(y_true, batch_size)
    probs = softmax(logits_2d)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(batch_size), y] = 1.0
    diff = probs - one_hot
    weighted = np.sum(diff * probs, axis=1, keepdims=True)
    dLogits = 2.0 * probs * (diff - weighted) / batch_size
    return dLogits[0] if was_1d else dLogits


# ── Dispatcher ────────────────────────────────────────────────────────────

def get_loss(name):
    """Return (loss_fn, gradient_fn) for the given loss name."""
    n = name.lower().replace("-", "_").replace(" ", "_")
    if n in ("cross_entropy", "ce", "cross_entropy_loss"):
        return cross_entropy_loss, cross_entropy_gradient
    elif n in ("mean_squared_error", "mse", "mse_loss"):
        return mse_loss, mse_gradient
    else:
        raise ValueError(f"Unknown loss: '{name}'. Choose: cross_entropy, mean_squared_error")
