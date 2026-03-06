"""
Gradient-based optimizers: SGD, Momentum, NAG, RMSProp.

Each optimizer exposes:
    update(layers) -> None
which reads layer.grad_W / layer.grad_b and modifies layer.W / layer.b in-place.
"""
import numpy as np


class SGD:
    def __init__(self, lr=0.01, momentum=0.0, weight_decay=0.0):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.v = {}

    def update(self, layers):
        for lid, layer in enumerate(layers):
            if lid not in self.v:
                self.v[lid] = {'W': np.zeros_like(layer.W),
                               'b': np.zeros_like(layer.b)}
            gW = layer.grad_W + self.weight_decay * layer.W
            gb = layer.grad_b
            self.v[lid]['W'] = self.momentum * self.v[lid]['W'] - self.lr * gW
            self.v[lid]['b'] = self.momentum * self.v[lid]['b'] - self.lr * gb
            layer.W += self.v[lid]['W']
            layer.b += self.v[lid]['b']


class NAG:
    """Nesterov Accelerated Gradient."""
    def __init__(self, lr=0.01, momentum=0.9, weight_decay=0.0):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.v = {}

    def update(self, layers):
        for lid, layer in enumerate(layers):
            if lid not in self.v:
                self.v[lid] = {'W': np.zeros_like(layer.W),
                               'b': np.zeros_like(layer.b)}
            gW = layer.grad_W + self.weight_decay * layer.W
            gb = layer.grad_b
            v_prev_W = self.v[lid]['W'].copy()
            v_prev_b = self.v[lid]['b'].copy()
            self.v[lid]['W'] = self.momentum * self.v[lid]['W'] - self.lr * gW
            self.v[lid]['b'] = self.momentum * self.v[lid]['b'] - self.lr * gb
            # Nesterov correction
            layer.W += -self.momentum * v_prev_W + (1 + self.momentum) * self.v[lid]['W']
            layer.b += -self.momentum * v_prev_b + (1 + self.momentum) * self.v[lid]['b']


class RMSProp:
    def __init__(self, lr=0.001, beta=0.9, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.weight_decay = weight_decay
        self.v = {}

    def update(self, layers):
        for lid, layer in enumerate(layers):
            if lid not in self.v:
                self.v[lid] = {'W': np.zeros_like(layer.W),
                               'b': np.zeros_like(layer.b)}
            gW = layer.grad_W + self.weight_decay * layer.W
            gb = layer.grad_b
            self.v[lid]['W'] = self.beta * self.v[lid]['W'] + (1 - self.beta) * gW ** 2
            self.v[lid]['b'] = self.beta * self.v[lid]['b'] + (1 - self.beta) * gb ** 2
            layer.W -= self.lr * gW / (np.sqrt(self.v[lid]['W']) + self.eps)
            layer.b -= self.lr * gb / (np.sqrt(self.v[lid]['b']) + self.eps)


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.wd = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, layers):
        self.t += 1
        for lid, layer in enumerate(layers):
            if lid not in self.m:
                self.m[lid] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}
                self.v[lid] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}
            for p, g in [('W', layer.grad_W + self.wd * layer.W), ('b', layer.grad_b)]:
                self.m[lid][p] = self.b1 * self.m[lid][p] + (1 - self.b1) * g
                self.v[lid][p] = self.b2 * self.v[lid][p] + (1 - self.b2) * g ** 2
                mh = self.m[lid][p] / (1 - self.b1 ** self.t)
                vh = self.v[lid][p] / (1 - self.b2 ** self.t)
                if p == 'W':
                    layer.W -= self.lr * mh / (np.sqrt(vh) + self.eps)
                else:
                    layer.b -= self.lr * mh / (np.sqrt(vh) + self.eps)


class NAdam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.wd = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, layers):
        self.t += 1
        for lid, layer in enumerate(layers):
            if lid not in self.m:
                self.m[lid] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}
                self.v[lid] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}
            for p, g in [('W', layer.grad_W + self.wd * layer.W), ('b', layer.grad_b)]:
                self.m[lid][p] = self.b1 * self.m[lid][p] + (1 - self.b1) * g
                self.v[lid][p] = self.b2 * self.v[lid][p] + (1 - self.b2) * g ** 2
                mh = self.m[lid][p] / (1 - self.b1 ** self.t)
                vh = self.v[lid][p] / (1 - self.b2 ** self.t)
                nadam_update = (self.b1 * mh + (1 - self.b1) * g / (1 - self.b1 ** self.t)) / (np.sqrt(vh) + self.eps)
                if p == 'W':
                    layer.W -= self.lr * nadam_update
                else:
                    layer.b -= self.lr * nadam_update


def get_optimizer(name, lr=0.001, weight_decay=0.0, **kwargs):
    """
    Return an optimizer instance.
    name: 'sgd' | 'momentum' | 'nag' | 'nesterov' | 'rmsprop' | 'adam' | 'nadam'
    """
    n = name.lower().strip()
    if n == 'sgd':
        return SGD(lr=lr, momentum=0.0, weight_decay=weight_decay)
    elif n in ('momentum', 'mgd'):
        return SGD(lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif n in ('nag', 'nesterov'):
        return NAG(lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif n == 'rmsprop':
        return RMSProp(lr=lr, weight_decay=weight_decay)
    elif n == 'adam':
        return Adam(lr=lr, weight_decay=weight_decay)
    elif n == 'nadam':
        return NAdam(lr=lr, weight_decay=weight_decay)
    else:
        # Fallback to Adam for unknown names rather than returning None
        print(f"Warning: unknown optimizer '{name}', defaulting to Adam.")
        return Adam(lr=lr, weight_decay=weight_decay)
