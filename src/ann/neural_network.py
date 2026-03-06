"""
NeuralNetwork: a configurable multi-layer perceptron.

Key design decisions (per autograder requirements):
- forward() returns raw logits (no softmax).
- backward() propagates from last layer to first;
  each layer exposes self.grad_W and self.grad_b.
- get_weights() / set_weights() for model serialisation.
- NeuralNetwork() accepts an argparse.Namespace, a dict, OR plain keyword args.
"""
import argparse
import numpy as np
from .neural_layer import NeuralLayer
from .activations import get_activation, softmax
from .objective_functions import get_loss
from .optimizers import get_optimizer


class NeuralNetwork:
    """
    Configurable fully-connected MLP.

    Can be constructed three ways:
        NeuralNetwork(args)                          # argparse.Namespace
        NeuralNetwork({'num_layers': 2, ...})        # plain dict
        NeuralNetwork(input_size=784, hidden_sizes=[128,128], activation='relu', ...)
    """

    def __init__(self, args=None, input_size=784, output_size=10,
                 hidden_sizes=None, hidden_size=None, num_layers=None,
                 activation='relu', weight_init='xavier', loss='cross_entropy',
                 weight_decay=0.0, **kwargs):

        # ── Resolve configuration from whatever was passed ─────────────────
        if isinstance(args, argparse.Namespace):
            cfg = vars(args)
        elif isinstance(args, dict):
            cfg = args
        else:
            # Called with plain keyword args (args is None or a positional int)
            cfg = {}

        def _get(key, default):
            # keyword arg → cfg dict → fallback default
            return kwargs.get(key, cfg.get(key, default))

        self.input_size   = int(_get('input_size',  input_size))
        self.output_size  = int(_get('output_size', output_size))
        self.activation   = _get('activation',  activation)
        self.weight_init  = _get('weight_init', weight_init)
        self.loss_name    = _get('loss',         loss)
        self.weight_decay = float(_get('weight_decay', weight_decay))

        _num_layers  = _get('num_layers',  num_layers)
        _hidden_size = _get('hidden_size', hidden_size)
        _hidden_sizes = _get('hidden_sizes', hidden_sizes)

        # ── Build hidden_sizes list ─────────────────────────────────────────
        if _hidden_sizes is not None:
            if isinstance(_hidden_sizes, (int, np.integer)):
                _hidden_sizes = [int(_hidden_sizes)]
            hidden = [int(h) for h in _hidden_sizes]
        elif _hidden_size is not None and _num_layers is not None:
            hidden = [int(_hidden_size)] * int(_num_layers)
        elif _hidden_size is not None:
            hidden = [int(_hidden_size)]
        elif _num_layers is not None:
            hidden = [128] * int(_num_layers)
        else:
            hidden = [128]

        self.hidden_sizes = hidden

        # ── Build layers ────────────────────────────────────────────────────
        self.layers = []
        sizes = [self.input_size] + hidden + [self.output_size]
        for i in range(len(sizes) - 1):
            is_output = (i == len(sizes) - 2)
            act = 'linear' if is_output else self.activation
            self.layers.append(
                NeuralLayer(sizes[i], sizes[i + 1],
                            activation=act,
                            weight_init=self.weight_init,
                            is_output=is_output)
            )

        # ── Loss function ───────────────────────────────────────────────────
        self.loss_fn, self.loss_grad_fn = get_loss(self.loss_name)

    # ───────────────────────────────────────────────────────────────────────
    # Forward pass
    # ───────────────────────────────────────────────────────────────────────

    def forward(self, X):
        """
        Forward pass through all layers.
        Returns raw logits (no softmax), shape (batch, output_size).
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    # ───────────────────────────────────────────────────────────────────────
    # Backward pass
    # ───────────────────────────────────────────────────────────────────────

    def backward(self, logits, y_true):
        """
        Compute and store gradients from last layer to first.

        Parameters
        ----------
        logits : (batch, output_size) – raw outputs from forward()
        y_true : (batch,) integer labels

        Returns
        -------
        grad_Ws, grad_bs – lists ordered layer 0 → last
        """
        dA = self.loss_grad_fn(logits, y_true)
        for layer in reversed(self.layers):
            dA = layer.backward(dA, weight_decay=self.weight_decay)

        grad_W = [l.grad_W for l in self.layers]
        grad_b = [l.grad_b for l in self.layers]
        return grad_W, grad_b

    # ───────────────────────────────────────────────────────────────────────
    # Prediction helpers
    # ───────────────────────────────────────────────────────────────────────

    def predict_proba(self, X):
        return softmax(self.forward(X))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def compute_loss(self, X, y):
        return self.loss_fn(self.forward(X), y)

    def compute_accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))

    # ───────────────────────────────────────────────────────────────────────
    # Weight serialisation  (required by autograder)
    # ───────────────────────────────────────────────────────────────────────

    def get_weights(self):
        """
        Return a flat dict suitable for np.save / np.load.
        Keys: 'W0', 'b0', 'W1', 'b1', ...
        """
        d = {}
        for i, layer in enumerate(self.layers):
            d[f'W{i}'] = layer.W.copy()
            d[f'b{i}'] = layer.b.copy()
        return d

    def set_weights(self, weights):
        """
        Load from a dict  {'W0': ..., 'b0': ..., ...}
        or from np.load(..., allow_pickle=True).item()
        """
        if isinstance(weights, np.ndarray) and weights.ndim == 0:
            weights = weights.item()
        if isinstance(weights, dict):
            for i, layer in enumerate(self.layers):
                if f'W{i}' in weights:
                    layer.W = np.array(weights[f'W{i}']).copy()
                if f'b{i}' in weights:
                    layer.b = np.array(weights[f'b{i}']).copy()
        else:
            # fallback: flat list [W0, b0, W1, b1, ...]
            weights = list(weights)
            for i, layer in enumerate(self.layers):
                layer.W = np.array(weights[2 * i]).copy()
                layer.b = np.array(weights[2 * i + 1]).copy()

    # ───────────────────────────────────────────────────────────────────────
    # Training loop
    # ───────────────────────────────────────────────────────────────────────

    def fit(self, X_train, y_train, X_val, y_val, args, wandb_run=None):
        """Full training loop with optional W&B logging."""
        cfg = vars(args) if isinstance(args, argparse.Namespace) else dict(args)

        epochs     = int(cfg.get('epochs',        10))
        batch_size = int(cfg.get('batch_size',    32))
        opt_name   = cfg.get('optimizer',         'rmsprop')
        lr         = float(cfg.get('learning_rate', 0.001))
        wd         = float(cfg.get('weight_decay',  0.0))

        optimizer = get_optimizer(opt_name, lr, wd)

        n = X_train.shape[0]
        best_val_acc = -1.0
        best_weights = None

        for epoch in range(1, epochs + 1):
            idx = np.random.permutation(n)
            X_shuf, y_shuf = X_train[idx], y_train[idx]

            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, n, batch_size):
                Xb = X_shuf[start: start + batch_size]
                yb = y_shuf[start: start + batch_size]

                logits = self.forward(Xb)
                epoch_loss += self.loss_fn(logits, yb)
                num_batches += 1

                self.backward(logits, yb)
                optimizer.update(self.layers)

            avg_loss  = epoch_loss / num_batches
            train_acc = self.compute_accuracy(X_train, y_train)
            val_acc   = self.compute_accuracy(X_val,   y_val)
            val_loss  = self.compute_loss(X_val, y_val)

            print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

            if wandb_run is not None:
                wandb_run.log({'epoch': epoch, 'train_loss': avg_loss,
                               'val_loss': val_loss, 'train_acc': train_acc,
                               'val_acc': val_acc})

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = self.get_weights()

        return best_weights, best_val_acc
