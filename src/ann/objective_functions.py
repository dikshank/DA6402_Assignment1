"""
Loss / objective functions and their gradients.

Both functions receive *logits* (raw pre-softmax values) from the network.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import numpy as np
from ann.activations import softmax


# ---------------------------------------------------------------------------
# Cross-Entropy Loss  (with built-in Softmax)
# ---------------------------------------------------------------------------

def cross_entropy_loss(logits, y_true):
    """
    Softmax + Cross-Entropy loss.

    Parameters
    ----------
    logits  : (batch_size, num_classes)  – raw network outputs
    y_true  : (batch_size,)              – integer class labels

    Returns
    -------
    loss    : scalar mean loss
    """
    probs = softmax(logits)
    batch_size = logits.shape[0]
    # Clip for numerical stability
    log_probs = np.log(np.clip(probs[np.arange(batch_size), y_true], 1e-12, 1.0))
    loss = -np.mean(log_probs)
    return loss


def cross_entropy_gradient(logits, y_true):
    """
    Gradient of Softmax + Cross-Entropy w.r.t. logits.

    dL/d(logits) = (softmax(logits) - one_hot(y_true))
    (divided by batch_size is handled in layer.backward)

    Returns
    -------
    dLogits : (batch_size, num_classes)
    """
    probs = softmax(logits)
    batch_size = logits.shape[0]
    dLogits = probs.copy()
    dLogits[np.arange(batch_size), y_true] -= 1.0
    # We do NOT divide by batch_size here; the layer backward does /batch_size
    # via its own mean, but actually we want consistent semantics:
    # divide here so the upstream gradient is already per-sample-averaged.
    dLogits /= batch_size
    return dLogits


# ---------------------------------------------------------------------------
# Mean Squared Error  (with built-in Softmax for probability targets)
# ---------------------------------------------------------------------------

def mse_loss(logits, y_true):
    """
    MSE between softmax(logits) and one-hot(y_true).

    Parameters
    ----------
    logits  : (batch_size, num_classes)
    y_true  : (batch_size,) integer class labels

    Returns
    -------
    loss    : scalar
    """
    probs = softmax(logits)
    batch_size, num_classes = logits.shape
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(batch_size), y_true] = 1.0
    loss = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    return loss


def mse_gradient(logits, y_true):
    """
    Gradient of MSE(softmax(logits), one_hot) w.r.t. logits.

    Using the chain rule:
    dL/dz_i = (2/N) * sum_k [ (p_k - t_k) * p_k * (delta_ik - p_i) ]
    where p = softmax(z), t = one_hot(y_true).
    """
    probs = softmax(logits)
    batch_size, num_classes = logits.shape
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(batch_size), y_true] = 1.0

    diff = probs - one_hot  # (batch_size, num_classes)

    # Jacobian of softmax: dSoftmax_i/dz_j = p_i*(delta_ij - p_j)
    # dL/dz_j = sum_i [ (2*(p_i - t_i)) * p_i * (delta_ij - p_j) ]
    #          = 2 * p_j * (diff_j - sum_i(diff_i * p_i))
    weighted = np.sum(diff * probs, axis=1, keepdims=True)  # (batch_size, 1)
    dLogits = 2.0 * probs * (diff - weighted)
    dLogits /= batch_size
    return dLogits


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def get_loss(name):
    """Return (loss_fn, gradient_fn) for the given loss name."""
    name = name.lower().replace("-", "_").replace(" ", "_")
    if name in ("cross_entropy", "ce", "cross-entropy"):
        return cross_entropy_loss, cross_entropy_gradient
    elif name in ("mean_squared_error", "mse"):
        return mse_loss, mse_gradient
    else:
        raise ValueError(f"Unknown loss: {name}. Choose 'cross_entropy' or 'mean_squared_error'.")
