"""
Activation functions and their derivatives for the MLP.
"""
import numpy as np


def sigmoid(z):
    """Sigmoid activation: 1 / (1 + exp(-z))"""
    # Clip for numerical stability
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    """Derivative of sigmoid: sigmoid(z) * (1 - sigmoid(z))"""
    s = sigmoid(z)
    return s * (1.0 - s)


def tanh(z):
    """Tanh activation: (exp(z) - exp(-z)) / (exp(z) + exp(-z))"""
    return np.tanh(z)


def tanh_derivative(z):
    """Derivative of tanh: 1 - tanh(z)^2"""
    return 1.0 - np.tanh(z) ** 2


def relu(z):
    """ReLU activation: max(0, z)"""
    return np.maximum(0.0, z)


def relu_derivative(z):
    """Derivative of ReLU: 1 if z > 0 else 0"""
    return (z > 0).astype(float)


def softmax(z):
    """
    Softmax activation for output layer.
    z: (batch_size, num_classes)
    """
    # Subtract max for numerical stability
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
        raise ValueError(f"Unknown activation function: {name}. Choose from: sigmoid, tanh, relu")
