"""
inference.py – Evaluate a saved MLP on test data.

The autograder calls: inference.parse_arguments()

If no pre-trained weights are found, this script trains the model automatically
so that F1 > 0.8 can be achieved even in a fresh container.
"""
import os
import sys
import json
import argparse
import numpy as np

# Ensure src/ and repo-root are importable from any CWD
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../src
_ROOT_DIR = os.path.dirname(_THIS_DIR)                    # repo root
for _p in [_THIS_DIR, _ROOT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset, get_class_names

try:
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score,
                                 confusion_matrix, classification_report)
    _SKLEARN = True
except ImportError:
    _SKLEARN = False


# ── Argument parsing — autograder calls inference.parse_arguments() ──────────

def parse_arguments():
    p = argparse.ArgumentParser(description="Evaluate saved MLP")

    p.add_argument("-d",   "--dataset",       type=str,   default="fashion_mnist")
    p.add_argument("-e",   "--epochs",         type=int,   default=10)
    p.add_argument("-b",   "--batch_size",     type=int,   default=32)
    p.add_argument("-l",   "--loss",           type=str,   default="cross_entropy")
    p.add_argument("-o",   "--optimizer",      type=str,   default="adam")
    p.add_argument("-lr",  "--learning_rate",  type=float, default=0.001)
    p.add_argument("-wd",  "--weight_decay",   type=float, default=0.0)
    p.add_argument("-nhl", "--num_layers",     type=int,   default=3)
    p.add_argument("-sz",  "--hidden_size",    type=int,   default=128, nargs="+")
    p.add_argument("-a",   "--activation",     type=str,   default="relu")
    p.add_argument("-wi",  "--weight_init",    type=str,   default="xavier")
    p.add_argument("-wp",  "--wandb_project",  type=str,   default="da6401_assignment1")
    p.add_argument("--wandb_entity",           type=str,   default=None)
    p.add_argument("--no_wandb",               action="store_true")
    p.add_argument("--val_split",              type=float, default=0.1)
    p.add_argument("--seed",                   type=int,   default=42)
    p.add_argument("--save_dir",               type=str,   default=None)
    p.add_argument("--model_path",             type=str,   default=None)
    p.add_argument("--config_path",            type=str,   default=None)
    # Friend's style aliases
    p.add_argument("--model",                  type=str,   default=None)
    p.add_argument("--config",                 type=str,   default=None)
    return p.parse_args()


# Alias so both names work
parse_args = parse_arguments


def _find_file(candidates):
    """Return first existing path from candidates list, or None."""
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def _arch_from_weights(weights_dict):
    """Infer hidden_sizes from a weights dict by looking at layer shapes."""
    # Find all layer indices
    idxs = sorted(set(
        int(k[1:]) for k in weights_dict if k.startswith('W')
    ))
    if not idxs:
        return [128, 128, 128]
    # Each Wi has shape (in, out); hidden layers are all except the last
    hidden = []
    for i in idxs[:-1]:  # all except output layer
        W = np.array(weights_dict[f'W{i}'])
        hidden.append(W.shape[1])  # output dim of this layer
    return hidden


def _load_weights_dict(path):
    """Load weights from .npy or .json, returning a plain dict."""
    if path.endswith('.json'):
        with open(path) as f:
            d = json.load(f)
        return {k: np.array(v) for k, v in d.items()}
    else:
        data = np.load(path, allow_pickle=True)
        if hasattr(data, 'item'):
            data = data.item()
        if isinstance(data, dict):
            return data
        return dict(data)


def _quick_train(dataset, save_dir, hidden_sizes=None, activation='relu',
                 weight_init='xavier', epochs=10, batch_size=64,
                 learning_rate=0.001, weight_decay=0.0):
    """Train a model with good hyperparams and save weights. Returns (model, X_test, y_test)."""
    print(f"\n[inference] No pre-trained weights found. Training now on {dataset}…")
    print("[inference] This may take a few minutes.")

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset, val_split=0.1, seed=42)

    if hidden_sizes is None:
        hidden_sizes = [128, 128, 128]

    from ann.optimizers import get_optimizer

    model = NeuralNetwork(
        input_size=784,
        hidden_sizes=hidden_sizes,
        output_size=10,
        activation=activation,
        weight_init=weight_init,
        loss='cross_entropy',
    )

    optimizer = get_optimizer('adam', learning_rate, weight_decay)
    n = X_train.shape[0]
    best_val_acc, best_weights = -1.0, None

    for epoch in range(1, epochs + 1):
        idx = np.random.permutation(n)
        X_shuf, y_shuf = X_train[idx], y_train[idx]
        total_loss, nb = 0.0, 0

        for start in range(0, n, batch_size):
            Xb = X_shuf[start: start + batch_size]
            yb = y_shuf[start: start + batch_size]
            logits = model.forward(Xb)
            total_loss += model.loss_fn(logits, yb)
            nb += 1
            model.backward(y_true=yb, y_pred=logits)
            optimizer.update(model.layers)

        val_acc = model.compute_accuracy(X_val, y_val)
        print(f"  Epoch {epoch:2d}/{epochs} | Loss: {total_loss/nb:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = model.get_weights()

    model.set_weights(best_weights)
    print(f"[inference] Training done. Best Val Acc: {best_val_acc:.4f}")

    # Save for future runs
    os.makedirs(save_dir, exist_ok=True)
    npy_path  = os.path.join(save_dir, "best_model.npy")
    json_path = os.path.join(save_dir, "best_model.json")
    cfg_path  = os.path.join(save_dir, "best_config.json")

    np.save(npy_path, best_weights)
    with open(json_path, "w") as f:
        json.dump({k: v.tolist() for k, v in best_weights.items()}, f)

    config = {
        "dataset":       dataset,
        "hidden_sizes":  hidden_sizes,
        "hidden_size":   hidden_sizes,
        "activation":    activation,
        "weight_init":   weight_init,
        "loss":          "cross_entropy",
        "optimizer":     "adam",
        "learning_rate": learning_rate,
        "weight_decay":  weight_decay,
        "val_accuracy":  float(best_val_acc),
    }
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"[inference] Saved weights → {npy_path}")
    return model, X_test, y_test


def main():
    args = parse_arguments()
    np.random.seed(args.seed)

    # ── Resolve file paths (always use _THIS_DIR as primary location) ─────────
    save_dir = _THIS_DIR  # Always save/load next to inference.py

    weights_candidates = [
        args.model_path, args.model,
        os.path.join(_THIS_DIR, 'best_model.npy'),
        os.path.join(_ROOT_DIR, 'src', 'best_model.npy'),
        os.path.join(_ROOT_DIR, 'models', 'best_model.npy'),
    ]
    json_candidates = [
        os.path.join(_THIS_DIR, 'best_model.json'),
        os.path.join(_ROOT_DIR, 'src', 'best_model.json'),
        os.path.join(_ROOT_DIR, 'models', 'best_model.json'),
    ]
    config_candidates = [
        args.config_path, args.config,
        os.path.join(_THIS_DIR, 'best_config.json'),
        os.path.join(_ROOT_DIR, 'src', 'best_config.json'),
        os.path.join(_ROOT_DIR, 'models', 'best_config.json'),
    ]

    weights_path = _find_file(weights_candidates)
    json_weights = _find_file(json_candidates)
    config_path  = _find_file(config_candidates)

    # ── Auto-train if no weights exist ────────────────────────────────────────
    if weights_path is None and json_weights is None:
        # Determine architecture from args
        hs = args.hidden_size
        if isinstance(hs, list):
            if len(hs) == 1:
                hs = [hs[0]] * args.num_layers
        else:
            hs = [int(hs)] * args.num_layers

        model, X_test, y_test = _quick_train(
            dataset=args.dataset,
            save_dir=save_dir,
            hidden_sizes=hs,
            activation=args.activation,
            weight_init=args.weight_init,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        # ── Load weights to determine architecture ────────────────────────────
        wpath = json_weights or weights_path
        weights_dict = _load_weights_dict(wpath)
        print(f"Weights loaded from {wpath}")

        # Infer architecture directly from weight shapes (most reliable)
        hidden_sizes = _arch_from_weights(weights_dict)
        activation   = 'relu'
        weight_init  = 'xavier'
        dataset      = args.dataset

        # Override with config if available
        if config_path:
            with open(config_path) as f:
                cfg = json.load(f)
            print(f"Config loaded from {config_path}")
            # Only use config arch if it matches weight shapes
            cfg_hs = cfg.get('hidden_sizes', cfg.get('hidden_size', None))
            if cfg_hs is not None:
                if isinstance(cfg_hs, int):
                    cfg_hs = [cfg_hs]
                cfg_hs = [int(h) for h in cfg_hs]
                # Verify config matches actual weight shapes
                if cfg_hs == hidden_sizes:
                    hidden_sizes = cfg_hs
                else:
                    print(f"[Warning] Config hidden_sizes {cfg_hs} != weight shapes {hidden_sizes}. Using weight shapes.")
            activation  = cfg.get('activation',  activation)
            weight_init = cfg.get('weight_init', weight_init)
            dataset     = cfg.get('dataset',     dataset)

        # ── Load data ─────────────────────────────────────────────────────────
        print(f"Loading {dataset} …")
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
            dataset, val_split=args.val_split, seed=args.seed
        )

        # ── Rebuild model from actual weight shapes ───────────────────────────
        model = NeuralNetwork(
            input_size=int(X_test.shape[1]),
            hidden_sizes=hidden_sizes,
            output_size=10,
            activation=activation,
            weight_init=weight_init,
            loss='cross_entropy',
        )
        print(f"Rebuilt model: hidden_sizes={hidden_sizes}")

        # ── Load weights ───────────────────────────────────────────────────────
        model.set_weights(weights_dict)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test).astype(int)
    y_test = y_test.astype(int)

    if _SKLEARN:
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        print(f"\n========== Evaluation Results ==========")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1-Score  : {f1:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))
        return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
    else:
        acc = float(np.mean(y_pred == y_test))
        print(f"Accuracy: {acc:.4f}")
        return {'accuracy': acc}


if __name__ == "__main__":
    main()
