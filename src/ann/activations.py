"""
Activation functions and their derivatives for the MLP.
"""
import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1.0 - s)


def tanh(z):
    return np.tanh(z)


def tanh_derivative(z):
    return 1.0 - np.tanh(z) ** 2


def relu(z):
    return np.maximum(0.0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def softmax(z):
    """
    Numerically stable softmax.
    Accepts 1D (num_classes,) or 2D (batch, num_classes) input.
    Always returns same shape as input.
    """
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z_shifted = z - np.max(z)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z)
    else:
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def get_activation(name):
    """Return (activation_fn, derivative_fn) for the given name."""
    name = name.lower()
    if name == "sigmoid":
        return sigmoid, sigmoid_derivative
    elif name == "tanh":
        return tanh, tanh_derivative
    elif name == "relu":
        return relu, relu_derivative
    else:
        raise ValueError(f"Unknown activation: '{name}'. Choose: sigmoid, tanh, relu")
