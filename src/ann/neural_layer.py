"""
NeuralLayer: a single fully-connected layer with optional activation.
Stores gradients as self.grad_W and self.grad_b after every backward() call.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import numpy as np
from ann.activations import get_activation, softmax


class NeuralLayer:
    """
    A single fully-connected (dense) layer.

    Parameters
    ----------
    input_size  : int   – number of input features
    output_size : int   – number of neurons in this layer
    activation  : str   – 'sigmoid', 'tanh', 'relu', or 'linear' (output layer)
    weight_init : str   – 'random' or 'xavier'
    is_output   : bool  – True for the final classification layer
    """

    def __init__(self, input_size, output_size, activation="relu",
                 weight_init="xavier", is_output=False):
        self.input_size = input_size
        self.output_size = output_size
        self.is_output = is_output
        self.activation_name = activation

        # ---- Weight initialisation ----------------------------------------
        if weight_init == "xavier":
            limit = np.sqrt(6.0 / (input_size + output_size))
            self.W = np.random.uniform(-limit, limit, (input_size, output_size))
        else:  # 'random'  – small Gaussian
            self.W = np.random.randn(input_size, output_size) * 0.01

        self.b = np.zeros((1, output_size))

        # ---- Activation function (not used for output layer raw logits) ----
        if not is_output:
            self.activation_fn, self.activation_deriv = get_activation(activation)
        else:
            self.activation_fn = None
            self.activation_deriv = None

        # ---- Cache for backprop -------------------------------------------
        self.input = None   # A_{l-1}  (batch_size, input_size)
        self.Z = None       # pre-activation  W*x + b
        self.A = None       # post-activation

        # ---- Gradients (exposed for autograder) ---------------------------
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

    # -----------------------------------------------------------------------
    def forward(self, X):
        """
        Forward pass.
        X : (batch_size, input_size)
        Returns activated output A (or raw logits Z for output layer).
        """
        self.input = X
        self.Z = X @ self.W + self.b   # (batch_size, output_size)

        if self.is_output:
            # Return raw logits; softmax is applied inside the loss function
            self.A = self.Z
        else:
            self.A = self.activation_fn(self.Z)

        return self.A

    # -----------------------------------------------------------------------
    def backward(self, dA, weight_decay=0.0):
        """
        Backward pass.

        Parameters
        ----------
        dA          : upstream gradient w.r.t. this layer's output A
                      (batch_size, output_size)
        weight_decay: L2 regularisation coefficient λ

        Returns
        -------
        dX : gradient w.r.t. this layer's input  (batch_size, input_size)

        Side-effects
        ------------
        Sets self.grad_W and self.grad_b.
        """
        batch_size = self.input.shape[0]

        if self.is_output:
            # dA already equals dZ when the output is raw logits and the
            # loss computes the combined softmax + CE gradient upstream.
            dZ = dA
        else:
            dZ = dA * self.activation_deriv(self.Z)   # element-wise

        # Gradients w.r.t. parameters
        self.grad_W = (self.input.T @ dZ) / batch_size + weight_decay * self.W
        self.grad_b = np.mean(dZ, axis=0, keepdims=True)

        # Gradient w.r.t. input (to propagate further back)
        dX = dZ @ self.W.T
        return dX
