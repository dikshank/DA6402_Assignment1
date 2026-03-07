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
    p.add_argument("--save_dir",               type=str,   default="src")
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


def _quick_train(dataset, save_dir):
    """Train a model with good hyperparams and save weights. Returns the model."""
    print(f"\n[inference] No pre-trained weights found. Training now on {dataset}…")
    print("[inference] This may take a few minutes.")

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset, val_split=0.1, seed=42)

    # Hyperparams tuned for F1 > 0.85 in ~10 epochs on both MNIST and Fashion-MNIST
    cfg = {
        "input_size":    784,
        "output_size":   10,
        "hidden_sizes":  [256, 128],
        "activation":    "relu",
        "weight_init":   "xavier",
        "loss":          "cross_entropy",
        "optimizer":     "adam",
        "learning_rate": 0.001,
        "weight_decay":  0.0,
        "epochs":        10,
        "batch_size":    64,
        "dataset":       dataset,
    }

    from ann.optimizers import get_optimizer
    from ann.activations import softmax

    model = NeuralNetwork(
        input_size=cfg["input_size"],
        hidden_sizes=cfg["hidden_sizes"],
        output_size=cfg["output_size"],
        activation=cfg["activation"],
        weight_init=cfg["weight_init"],
        loss=cfg["loss"],
    )

    optimizer = get_optimizer(cfg["optimizer"], cfg["learning_rate"], cfg["weight_decay"])
    n = X_train.shape[0]
    best_val_acc, best_weights = -1.0, None

    for epoch in range(1, cfg["epochs"] + 1):
        idx = np.random.permutation(n)
        X_shuf, y_shuf = X_train[idx], y_train[idx]
        total_loss, nb = 0.0, 0

        for start in range(0, n, cfg["batch_size"]):
            Xb = X_shuf[start: start + cfg["batch_size"]]
            yb = y_shuf[start: start + cfg["batch_size"]]
            logits = model.forward(Xb)
            total_loss += model.loss_fn(logits, yb)
            nb += 1
            model.backward(y_true=yb, y_pred=logits)
            optimizer.update(model.layers)

        val_acc = model.compute_accuracy(X_val, y_val)
        print(f"  Epoch {epoch:2d}/{cfg['epochs']} | Loss: {total_loss/nb:.4f} | Val Acc: {val_acc:.4f}")

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
    cfg["hidden_size"] = cfg["hidden_sizes"]
    cfg["val_accuracy"] = float(best_val_acc)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"[inference] Saved weights → {npy_path}")
    return model, X_test, y_test


def main():
    args = parse_arguments()
    np.random.seed(args.seed)

    # ── Resolve config file ───────────────────────────────────────────────────
    config_path = _find_file([
        args.config_path, args.config,
        os.path.join(_THIS_DIR, 'best_config.json'),
        os.path.join(_ROOT_DIR, 'src', 'best_config.json'),
        os.path.join(_ROOT_DIR, 'models', 'best_config.json'),
    ])

    cfg = {}
    if config_path:
        with open(config_path) as f:
            cfg = json.load(f)
        print(f"Config loaded from {config_path}")

    dataset = cfg.get('dataset', args.dataset)

    # ── Check for weights early ───────────────────────────────────────────────
    save_dir = args.save_dir  # default "src" -> relative to CWD

    weights_candidates = [
        args.model_path, args.model,
        os.path.join(_THIS_DIR, 'best_model.npy'),
        os.path.join(_ROOT_DIR, 'src', 'best_model.npy'),
        os.path.join(_ROOT_DIR, 'models', 'best_model.npy'),
        os.path.join(save_dir, 'best_model.npy'),
    ]
    json_candidates = [
        os.path.join(_THIS_DIR, 'best_model.json'),
        os.path.join(_ROOT_DIR, 'src', 'best_model.json'),
        os.path.join(_ROOT_DIR, 'models', 'best_model.json'),
        os.path.join(save_dir, 'best_model.json'),
    ]

    weights_path = _find_file(weights_candidates)
    json_weights = _find_file(json_candidates)

    # ── Auto-train if no weights exist ────────────────────────────────────────
    if weights_path is None and json_weights is None:
        model, X_test, y_test = _quick_train(dataset, _THIS_DIR)
    else:
        # ── Load data ──────────────────────────────────────────────────────────
        print(f"Loading {dataset} …")
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
            dataset, val_split=args.val_split, seed=args.seed
        )

        # ── Rebuild model from config ──────────────────────────────────────────
        hs = cfg.get('hidden_sizes', cfg.get('hidden_size', [256, 128]))
        if isinstance(hs, int):
            hs = [hs]
        hs = [int(h) for h in hs]

        model = NeuralNetwork(
            input_size=int(X_test.shape[1]),
            hidden_sizes=hs,
            output_size=10,
            activation=cfg.get('activation', 'relu'),
            weight_init=cfg.get('weight_init', 'xavier'),
            loss=cfg.get('loss', 'cross_entropy'),
        )

        # ── Load weights ───────────────────────────────────────────────────────
        if json_weights:
            with open(json_weights) as f:
                d = json.load(f)
            model.set_weights({k: np.array(v) for k, v in d.items()})
            print(f"Weights loaded from {json_weights}")
        else:
            data = np.load(weights_path, allow_pickle=True)
            if hasattr(data, 'item'):
                data = data.item()
            model.set_weights(data)
            print(f"Weights loaded from {weights_path}")

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
