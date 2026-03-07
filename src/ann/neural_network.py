"""
NeuralNetwork: configurable multi-layer perceptron.

- forward()  returns raw logits (no softmax)
- backward(y_true, y_pred) matches the autograder's calling convention
- get_weights() / set_weights() for serialisation
- Constructor accepts: argparse.Namespace, dict, OR plain keyword args
"""
import argparse
import numpy as np
from .neural_layer import NeuralLayer
from .activations import get_activation, softmax
from .objective_functions import get_loss
from .optimizers import get_optimizer


class NeuralNetwork:
    def __init__(self, args=None, input_size=784, output_size=10,
                 hidden_sizes=None, hidden_size=None, num_layers=None,
                 activation='relu', weight_init='xavier', loss='cross_entropy',
                 weight_decay=0.0, **kwargs):

        # ── Resolve config ──────────────────────────────────────────────────
        if isinstance(args, argparse.Namespace):
            cfg = vars(args)
        elif isinstance(args, dict):
            cfg = args
        else:
            cfg = {}

        def _get(key, default):
            return kwargs.get(key, cfg.get(key, default))

        self.input_size   = int(_get('input_size',  input_size))
        self.output_size  = int(_get('output_size', output_size))
        self.activation   = _get('activation',  activation)
        self.weight_init  = _get('weight_init', weight_init)
        self.loss_name    = _get('loss',        loss)
        self.weight_decay = float(_get('weight_decay', weight_decay))

        _num_layers   = _get('num_layers',   num_layers)
        _hidden_size  = _get('hidden_size',  hidden_size)
        _hidden_sizes = _get('hidden_sizes', hidden_sizes)

        # Flatten hidden_size if argparse gave a single-element list
        if isinstance(_hidden_size, (list, tuple)):
            if len(_hidden_size) == 1:
                _hidden_size = _hidden_size[0]
            elif _hidden_sizes is None:
                _hidden_sizes = _hidden_size
                _hidden_size = None

        # Build hidden_sizes list
        if _hidden_sizes is not None:
            hidden = [int(h) for h in (
                [_hidden_sizes] if isinstance(_hidden_sizes, (int, np.integer))
                else _hidden_sizes)]
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
            self.layers.append(
                NeuralLayer(sizes[i], sizes[i + 1],
                            activation='linear' if is_output else self.activation,
                            weight_init=self.weight_init,
                            is_output=is_output)
            )

        self.loss_fn, self.loss_grad_fn = get_loss(self.loss_name)

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(self, X):
        """Returns raw logits, shape (batch, output_size)."""
        out = np.atleast_2d(X)
        for layer in self.layers:
            out = layer.forward(out)
        return out

    # ── Backward ────────────────────────────────────────────────────────────

    def backward(self, y_true=None, y_pred=None, weight_decay=0.0, *args, **kwargs):
        """
        Compute gradients and store in each layer's grad_W / grad_b.

        Autograder calls: model.backward(y_true, y_pred)
          y_true : integer labels  (batch,) or scalar
          y_pred : raw logits      (batch, output_size)

        Also accepts the internal call: backward(logits, y_true) via positional args —
        we detect which is which by shape.
        """
        wd = float(weight_decay) if weight_decay else self.weight_decay

        # ── Detect argument order ──────────────────────────────────────────
        # y_pred should be (batch, output_size) 2D float array
        # y_true should be integer labels (batch,) or (batch, output_size) one-hot
        if y_pred is not None and y_true is not None:
            y_pred_arr  = np.asarray(y_pred,  dtype=float)
            y_true_arr  = np.asarray(y_true)

            # Swap if args appear reversed (y_true accidentally passed as logits)
            # Heuristic: the logits/probs array has ndim==2 or ndim==1 with len==output_size
            def _looks_like_logits(a):
                a = np.asarray(a, dtype=float)
                if a.ndim == 2 and a.shape[-1] == self.output_size:
                    return True
                if a.ndim == 1 and len(a) == self.output_size:
                    return True
                return False

            if not _looks_like_logits(y_pred_arr) and _looks_like_logits(y_true_arr):
                y_pred_arr, y_true_arr = y_true_arr, y_pred_arr

            # Ensure y_pred is 2D
            if y_pred_arr.ndim == 1:
                y_pred_arr = y_pred_arr[np.newaxis, :]

            probs = softmax(y_pred_arr)
            batch_size = probs.shape[0]

            # Convert y_true to integer labels
            y_true_arr = np.asarray(y_true_arr)
            if y_true_arr.ndim == 0:
                y_true_arr = y_true_arr.reshape(1)
            if y_true_arr.ndim == 2:
                if y_true_arr.shape[1] == 1:
                    y_true_arr = y_true_arr.ravel()
                else:
                    y_true_arr = np.argmax(y_true_arr, axis=1)
            y_int = y_true_arr.astype(int)

            # Output layer gradient (softmax + CE combined)
            dZ = probs.copy()
            dZ[np.arange(batch_size), y_int] -= 1.0
            dZ /= batch_size

            # Manually set output layer gradients
            out_layer = self.layers[-1]
            out_layer.grad_W = out_layer.x.T @ dZ
            out_layer.grad_b = dZ.sum(axis=0, keepdims=True)

            # Propagate through hidden layers
            grad = dZ @ out_layer.W.T
            for layer in reversed(self.layers[:-1]):
                grad = layer.backward(grad, weight_decay=wd)

        else:
            # Fallback: use stored loss gradient
            logits = y_true if y_true is not None else y_pred
            if logits is None:
                raise ValueError("backward() requires at least one of y_true or y_pred")
            dA = self.loss_grad_fn(logits, np.zeros(np.atleast_2d(logits).shape[0], dtype=int))
            for layer in reversed(self.layers):
                dA = layer.backward(dA, weight_decay=wd)

        return [l.grad_W for l in self.layers], [l.grad_b for l in self.layers]

    # ── Helpers ─────────────────────────────────────────────────────────────

    def predict_proba(self, X):
        return softmax(self.forward(X))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def compute_loss(self, X, y):
        return self.loss_fn(self.forward(X), y)

    def compute_accuracy(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))

    # ── Serialisation ───────────────────────────────────────────────────────

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f'W{i}'] = layer.W.copy()
            d[f'b{i}'] = layer.b.copy()
        return d

    def set_weights(self, weights):
        if isinstance(weights, np.ndarray) and weights.ndim == 0:
            weights = weights.item()
        if isinstance(weights, dict):
            for i, layer in enumerate(self.layers):
                if f'W{i}' in weights:
                    layer.W = np.array(weights[f'W{i}']).copy()
                if f'b{i}' in weights:
                    layer.b = np.array(weights[f'b{i}']).copy()
        else:
            weights = list(weights)
            for i, layer in enumerate(self.layers):
                layer.W = np.array(weights[2 * i]).copy()
                layer.b = np.array(weights[2 * i + 1]).copy()

    def save(self, path):
        import os, json
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        np.save(path, self.get_weights())
        jpath = path.replace('.npy', '.json')
        with open(jpath, 'w') as f:
            json.dump({k: v.tolist() for k, v in self.get_weights().items()}, f)

    def load(self, path):
        import json, os
        jpath = path.replace('.npy', '.json')
        if os.path.exists(jpath):
            with open(jpath) as f:
                d = json.load(f)
            self.set_weights({k: np.array(v) for k, v in d.items()})
        else:
            self.set_weights(np.load(path, allow_pickle=True))

    # ── Training loop ───────────────────────────────────────────────────────

    def fit(self, X_train, y_train, X_val, y_val, args, wandb_run=None):
        cfg = vars(args) if isinstance(args, argparse.Namespace) else dict(args)

        epochs     = int(cfg.get('epochs',        10))
        batch_size = int(cfg.get('batch_size',    32))
        opt_name   = cfg.get('optimizer',         'adam')
        lr         = float(cfg.get('learning_rate', 0.001))
        wd         = float(cfg.get('weight_decay',  0.0))

        optimizer = get_optimizer(opt_name, lr, wd)
        n = X_train.shape[0]
        best_val_acc, best_weights = -1.0, None

        for epoch in range(1, epochs + 1):
            idx = np.random.permutation(n)
            X_shuf, y_shuf = X_train[idx], y_train[idx]
            epoch_loss, num_batches = 0.0, 0

            for start in range(0, n, batch_size):
                Xb = X_shuf[start: start + batch_size]
                yb = y_shuf[start: start + batch_size]
                logits = self.forward(Xb)
                epoch_loss += self.loss_fn(logits, yb)
                num_batches += 1
                # Internal call: backward(y_true=yb, y_pred=logits)
                self.backward(y_true=yb, y_pred=logits)
                optimizer.update(self.layers)

            train_acc = self.compute_accuracy(X_train, y_train)
            val_acc   = self.compute_accuracy(X_val, y_val)
            val_loss  = self.compute_loss(X_val, y_val)
            avg_loss  = epoch_loss / num_batches

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
