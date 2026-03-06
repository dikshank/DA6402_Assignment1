"""
Gradient-based optimizers: SGD, Momentum, NAG, RMSProp.

Each optimizer class exposes a single method:
    update(layers) -> None
which reads layer.grad_W / layer.grad_b and modifies layer.W / layer.b in-place.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseOptimizer:
    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        self.lr = learning_rate
        self.weight_decay = weight_decay

    def update(self, layers):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# SGD (mini-batch gradient descent)
# ---------------------------------------------------------------------------

class SGD(BaseOptimizer):
    """Simple (mini-batch) stochastic gradient descent."""

    def update(self, layers):
        for layer in layers:
            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b


# ---------------------------------------------------------------------------
# SGD with Momentum
# ---------------------------------------------------------------------------

class Momentum(BaseOptimizer):
    """SGD with classical momentum."""

    def __init__(self, learning_rate=0.01, weight_decay=0.0, beta=0.9):
        super().__init__(learning_rate, weight_decay)
        self.beta = beta
        self.v_W = {}   # velocity for W, keyed by layer id
        self.v_b = {}

    def update(self, layers):
        for i, layer in enumerate(layers):
            if i not in self.v_W:
                self.v_W[i] = np.zeros_like(layer.W)
                self.v_b[i] = np.zeros_like(layer.b)

            self.v_W[i] = self.beta * self.v_W[i] + self.lr * layer.grad_W
            self.v_b[i] = self.beta * self.v_b[i] + self.lr * layer.grad_b

            layer.W -= self.v_W[i]
            layer.b -= self.v_b[i]


# ---------------------------------------------------------------------------
# Nesterov Accelerated Gradient (NAG)
# ---------------------------------------------------------------------------

class NAG(BaseOptimizer):
    """
    Nesterov Accelerated Gradient.
    We use the 'look-ahead' formulation:
        v_t = beta * v_{t-1} + lr * grad(W - beta * v_{t-1})
    Because we cannot run forward/backward twice efficiently here, we use the
    standard approximation where the Nesterov correction is applied to the
    parameter update rather than recomputing gradients:
        v_t = beta * v_{t-1} + lr * grad_t
        W  -= beta * v_t + lr * grad_t   (equivalent look-ahead update)
    """

    def __init__(self, learning_rate=0.01, weight_decay=0.0, beta=0.9):
        super().__init__(learning_rate, weight_decay)
        self.beta = beta
        self.v_W = {}
        self.v_b = {}

    def update(self, layers):
        for i, layer in enumerate(layers):
            if i not in self.v_W:
                self.v_W[i] = np.zeros_like(layer.W)
                self.v_b[i] = np.zeros_like(layer.b)

            v_W_prev = self.v_W[i].copy()
            v_b_prev = self.v_b[i].copy()

            self.v_W[i] = self.beta * self.v_W[i] + self.lr * layer.grad_W
            self.v_b[i] = self.beta * self.v_b[i] + self.lr * layer.grad_b

            # Nesterov update: use current velocity and correction
            layer.W -= (1 + self.beta) * self.v_W[i] - self.beta * v_W_prev
            layer.b -= (1 + self.beta) * self.v_b[i] - self.beta * v_b_prev


# ---------------------------------------------------------------------------
# RMSProp
# ---------------------------------------------------------------------------

class RMSProp(BaseOptimizer):
    """RMSProp optimizer."""

    def __init__(self, learning_rate=0.001, weight_decay=0.0, beta=0.9, epsilon=1e-8):
        super().__init__(learning_rate, weight_decay)
        self.beta = beta
        self.epsilon = epsilon
        self.s_W = {}   # running average of squared gradients
        self.s_b = {}

    def update(self, layers):
        for i, layer in enumerate(layers):
            if i not in self.s_W:
                self.s_W[i] = np.zeros_like(layer.W)
                self.s_b[i] = np.zeros_like(layer.b)

            self.s_W[i] = self.beta * self.s_W[i] + (1 - self.beta) * (layer.grad_W ** 2)
            self.s_b[i] = self.beta * self.s_b[i] + (1 - self.beta) * (layer.grad_b ** 2)

            layer.W -= self.lr * layer.grad_W / (np.sqrt(self.s_W[i]) + self.epsilon)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(self.s_b[i]) + self.epsilon)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def get_optimizer(name, learning_rate, weight_decay, **kwargs):
    """
    Factory function to return an optimizer instance.

    Parameters
    ----------
    name          : 'sgd', 'momentum', 'nag', 'rmsprop'
    learning_rate : float
    weight_decay  : float
    **kwargs      : additional optimizer-specific keyword arguments
    """
    name = name.lower()
    if name == "sgd":
        return SGD(learning_rate, weight_decay)
    elif name == "momentum":
        beta = kwargs.get("beta", 0.9)
        return Momentum(learning_rate, weight_decay, beta=beta)
    elif name == "nag":
        beta = kwargs.get("beta", 0.9)
        return NAG(learning_rate, weight_decay, beta=beta)
    elif name == "rmsprop":
        beta = kwargs.get("beta", 0.9)
        epsilon = kwargs.get("epsilon", 1e-8)
        return RMSProp(learning_rate, weight_decay, beta=beta, epsilon=epsilon)
    else:
        raise ValueError(f"Unknown optimizer: {name}. Choose from: sgd, momentum, nag, rmsprop")
