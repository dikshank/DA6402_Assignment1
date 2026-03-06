"""
NeuralLayer: a single fully-connected layer with optional activation.
Stores gradients as self.grad_W and self.grad_b after every backward() call.
"""
import numpy as np
from .activations import get_activation, softmax


class NeuralLayer:
    """
    A single fully-connected (dense) layer.

    Parameters
    ----------
    input_size  : int
    output_size : int
    activation  : str – 'sigmoid', 'tanh', 'relu', or 'linear' (output layer)
    weight_init : str – 'random' or 'xavier'
    is_output   : bool – True for the final classification layer
    """

    def __init__(self, input_size, output_size, activation='relu',
                 weight_init='xavier', is_output=False):
        self.input_size  = input_size
        self.output_size = output_size
        self.is_output   = is_output

        # ── Weight initialisation ───────────────────────────────────────────
        if weight_init == 'xavier':
            limit = np.sqrt(6.0 / (input_size + output_size))
            self.W = np.random.uniform(-limit, limit, (input_size, output_size))
        else:  # 'random'
            self.W = np.random.randn(input_size, output_size) * 0.01
        self.b = np.zeros((1, output_size))

        # ── Activation ──────────────────────────────────────────────────────
        act_name = activation if isinstance(activation, str) else 'linear'
        if act_name in ('linear', 'identity', 'none') or is_output:
            self.activation_fn    = None
            self.activation_deriv = None
        else:
            self.activation_fn, self.activation_deriv = get_activation(act_name)

        # ── Cache & gradients ───────────────────────────────────────────────
        self.x      = None   # input to this layer (alias: self.input)
        self.Z      = None   # pre-activation
        self.A      = None   # post-activation
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

    @property
    def input(self):
        return self.x

    @input.setter
    def input(self, val):
        self.x = val

    # ───────────────────────────────────────────────────────────────────────

    def forward(self, X):
        """
        X : (batch, input_size)
        Returns post-activation output (or raw logits for output layer).
        """
        self.x = X
        self.Z = X @ self.W + self.b

        if self.activation_fn is None:
            self.A = self.Z
        else:
            self.A = self.activation_fn(self.Z)

        return self.A

    # ───────────────────────────────────────────────────────────────────────

    def backward(self, dA, weight_decay=0.0):
        """
        dA : upstream gradient w.r.t. this layer's output, (batch, output_size)
        Returns dX : gradient w.r.t. input, (batch, input_size)
        Sets self.grad_W and self.grad_b.
        """
        batch_size = self.x.shape[0]

        if self.activation_deriv is None:
            dZ = dA
        else:
            dZ = dA * self.activation_deriv(self.Z)

        self.grad_W = (self.x.T @ dZ) / batch_size + weight_decay * self.W
        self.grad_b = np.mean(dZ, axis=0, keepdims=True)

        dX = dZ @ self.W.T
        return dX
