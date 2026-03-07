"""
NeuralLayer: a single fully-connected layer with optional activation.
Stores gradients as self.grad_W and self.grad_b after every backward() call.
"""
import numpy as np
from .activations import get_activation, softmax


class NeuralLayer:
    def __init__(self, input_size, output_size, activation='relu',
                 weight_init='xavier', is_output=False):
        self.input_size  = input_size
        self.output_size = output_size
        self.is_output   = is_output

        # Weight initialisation
        if weight_init == 'xavier':
            limit = np.sqrt(6.0 / (input_size + output_size))
            self.W = np.random.uniform(-limit, limit, (input_size, output_size))
        else:
            self.W = np.random.randn(input_size, output_size) * 0.01
        self.b = np.zeros((1, output_size))

        # Activation
        act_name = activation if isinstance(activation, str) else 'linear'
        if act_name in ('linear', 'identity', 'none') or is_output:
            self.activation_fn    = None
            self.activation_deriv = None
        else:
            self.activation_fn, self.activation_deriv = get_activation(act_name)

        # Cache & gradients
        self.x      = None
        self.Z      = None
        self.A      = None
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

    @property
    def input(self):
        return self.x

    @input.setter
    def input(self, val):
        self.x = val

    def forward(self, X):
        """X: (batch, input_size) or (input_size,). Returns post-activation output."""
        X = np.atleast_2d(X)
        self.x = X
        self.Z = X @ self.W + self.b   # (batch, output_size)
        self.A = self.Z if self.activation_fn is None else self.activation_fn(self.Z)
        return self.A

    def backward(self, dA, weight_decay=0.0):
        """
        dA: upstream gradient, (batch, output_size) or (output_size,).
        Returns dX: gradient w.r.t. input, same batch shape as dA.
        Sets self.grad_W and self.grad_b.
        """
        # Ensure 2D
        dA = np.atleast_2d(dA)
        batch_size = self.x.shape[0]

        if self.activation_deriv is None:
            dZ = dA
        else:
            dZ = dA * self.activation_deriv(self.Z)

        self.grad_W = (self.x.T @ dZ) / batch_size + weight_decay * self.W
        self.grad_b = np.mean(dZ, axis=0, keepdims=True)   # always (1, output_size)

        dX = dZ @ self.W.T
        return dX
