"""
NeuralNetwork: a configurable multi-layer perceptron.

Key design decisions (per autograder requirements):
- forward() returns raw logits (no softmax).
- backward() computes and stores gradients from last layer to first;
  each layer exposes self.grad_W and self.grad_b.
- get_weights() / set_weights() for model serialisation.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ann.neural_layer import NeuralLayer
from ann.activations import softmax
from ann.objective_functions import get_loss
from ann.optimizers import get_optimizer


class NeuralNetwork:
    """
    Configurable fully-connected MLP.

    Parameters (accepted via argparse Namespace or plain dict)
    ----------------------------------------------------------
    dataset        : str   – 'mnist' or 'fashion_mnist'
    epochs         : int
    batch_size     : int
    loss           : str   – 'cross_entropy' or 'mean_squared_error'
    optimizer      : str   – 'sgd' | 'momentum' | 'nag' | 'rmsprop'
    learning_rate  : float
    weight_decay   : float
    num_layers     : int   – number of *hidden* layers
    hidden_size    : int or list[int]  – neurons per hidden layer
    activation     : str   – 'sigmoid' | 'tanh' | 'relu'
    weight_init    : str   – 'random' | 'xavier'
    input_size     : int   – default 784 (28×28)
    output_size    : int   – default 10
    """

    def __init__(self, args, input_size=784, output_size=10):
        # Support both Namespace and dict
        if isinstance(args, dict):
            cfg = args
        else:
            cfg = vars(args)

        self.input_size  = cfg.get("input_size",  input_size)
        self.output_size = cfg.get("output_size", output_size)
        self.activation  = cfg.get("activation",  "relu")
        self.weight_init = cfg.get("weight_init", "xavier")
        self.loss_name   = cfg.get("loss",        "cross_entropy")
        self.weight_decay= cfg.get("weight_decay", 0.0)

        num_layers   = cfg.get("num_layers",  3)
        hidden_size  = cfg.get("hidden_size", 128)

        # hidden_size may be a list or a single int
        if isinstance(hidden_size, (list, tuple)):
            hidden_sizes = list(hidden_size)
            # Pad/trim to num_layers
            if len(hidden_sizes) < num_layers:
                hidden_sizes += [hidden_sizes[-1]] * (num_layers - len(hidden_sizes))
            hidden_sizes = hidden_sizes[:num_layers]
        else:
            hidden_sizes = [int(hidden_size)] * num_layers

        # ---- Build layers --------------------------------------------------
        self.layers = []
        prev_size = self.input_size

        for h_size in hidden_sizes:
            self.layers.append(
                NeuralLayer(prev_size, h_size,
                            activation=self.activation,
                            weight_init=self.weight_init,
                            is_output=False)
            )
            prev_size = h_size

        # Output layer – no activation, returns raw logits
        self.layers.append(
            NeuralLayer(prev_size, self.output_size,
                        activation="linear",
                        weight_init=self.weight_init,
                        is_output=True)
        )

        # ---- Loss function ------------------------------------------------
        self.loss_fn, self.loss_grad = get_loss(self.loss_name)

    # -----------------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------------

    def forward(self, X):
        """
        Forward pass through all layers.

        Parameters
        ----------
        X : (batch_size, input_size)  – flattened pixel values in [0, 1]

        Returns
        -------
        logits : (batch_size, output_size)  – raw pre-softmax values
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out  # raw logits

    # -----------------------------------------------------------------------
    # Backward pass
    # -----------------------------------------------------------------------

    def backward(self, logits, y_true):
        """
        Backward pass: compute gradients from last layer to first.

        Parameters
        ----------
        logits : (batch_size, output_size)
        y_true : (batch_size,) integer labels

        Returns
        -------
        List of (grad_W, grad_b) tuples from last layer to first layer.
        Also stores gradients in each layer's self.grad_W / self.grad_b.
        """
        # Gradient of loss w.r.t. logits
        dA = self.loss_grad(logits, y_true)

        grads = []
        for layer in reversed(self.layers):
            dA = layer.backward(dA, weight_decay=self.weight_decay)
            grads.append((layer.grad_W.copy(), layer.grad_b.copy()))

        return grads  # ordered: last layer first

    # -----------------------------------------------------------------------
    # Prediction helpers
    # -----------------------------------------------------------------------

    def predict_proba(self, X):
        """Return softmax probabilities."""
        logits = self.forward(X)
        return softmax(logits)

    def predict(self, X):
        """Return integer class predictions."""
        return np.argmax(self.predict_proba(X), axis=1)

    def compute_loss(self, X, y):
        """Compute scalar loss over a dataset."""
        logits = self.forward(X)
        return self.loss_fn(logits, y)

    def compute_accuracy(self, X, y):
        """Compute classification accuracy."""
        preds = self.predict(X)
        return np.mean(preds == y)

    # -----------------------------------------------------------------------
    # Weight serialisation  (required by autograder)
    # -----------------------------------------------------------------------

    def get_weights(self):
        """
        Return a dict of all layer weights suitable for np.save.

        Format: {'layer_{i}_W': ..., 'layer_{i}_b': ..., 'config': {...}}
        """
        weights = {}
        for i, layer in enumerate(self.layers):
            weights[f"layer_{i}_W"] = layer.W
            weights[f"layer_{i}_b"] = layer.b
        weights["config"] = {
            "num_layers":   len(self.layers) - 1,   # hidden layers only
            "hidden_sizes": [l.output_size for l in self.layers[:-1]],
            "activation":   self.activation,
            "weight_init":  self.weight_init,
            "loss":         self.loss_name,
            "input_size":   self.input_size,
            "output_size":  self.output_size,
        }
        return weights

    def set_weights(self, weights):
        """
        Load weights from a dict (as returned by np.load(..., allow_pickle=True).item()).
        """
        for i, layer in enumerate(self.layers):
            layer.W = weights[f"layer_{i}_W"]
            layer.b = weights[f"layer_{i}_b"]

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    def fit(self, X_train, y_train, X_val, y_val, args, wandb_run=None):
        """
        Full training loop with optional W&B logging.

        Parameters
        ----------
        X_train, y_train : training data
        X_val,   y_val   : validation data
        args             : argparse Namespace (or dict) with training config
        wandb_run        : optional active W&B run for logging
        """
        if isinstance(args, dict):
            cfg = args
        else:
            cfg = vars(args)

        epochs     = cfg.get("epochs",     10)
        batch_size = cfg.get("batch_size", 32)
        opt_name   = cfg.get("optimizer",  "rmsprop")
        lr         = cfg.get("learning_rate", 0.001)
        wd         = cfg.get("weight_decay",  0.0)

        optimizer = get_optimizer(opt_name, lr, wd)

        n = X_train.shape[0]
        best_val_acc = -1.0
        best_weights = None

        for epoch in range(1, epochs + 1):
            # Shuffle
            idx = np.random.permutation(n)
            X_shuf, y_shuf = X_train[idx], y_train[idx]

            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, n, batch_size):
                X_batch = X_shuf[start: start + batch_size]
                y_batch = y_shuf[start: start + batch_size]

                # Forward
                logits = self.forward(X_batch)
                batch_loss = self.loss_fn(logits, y_batch)
                epoch_loss += batch_loss
                num_batches += 1

                # Backward
                self.backward(logits, y_batch)

                # Optimizer step
                optimizer.update(self.layers)

            avg_loss = epoch_loss / num_batches
            train_acc = self.compute_accuracy(X_train, y_train)
            val_acc   = self.compute_accuracy(X_val,   y_val)
            val_loss  = self.compute_loss(X_val, y_val)

            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_acc:.4f}")

            if wandb_run is not None:
                wandb_run.log({
                    "epoch":      epoch,
                    "train_loss": avg_loss,
                    "val_loss":   val_loss,
                    "train_acc":  train_acc,
                    "val_acc":    val_acc,
                })

            # Track best model by validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = self.get_weights()

        return best_weights, best_val_acc
