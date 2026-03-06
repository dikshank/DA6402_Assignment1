"""
inference.py – Evaluate a saved MLP on test data.

Usage:
    python src/inference.py -d mnist --model_path src/best_model.npy \
                            --config_path src/best_config.json
"""
import os
import sys
import json
import argparse
import numpy as np

# Ensure src/ and repo-root are on the path regardless of CWD
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
for _p in [_THIS_DIR, _ROOT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset, get_class_names


def build_parser():
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
    p.add_argument("--model_path",             type=str,   default="src/best_model.npy")
    p.add_argument("--config_path",            type=str,   default="src/best_config.json")
    return p


def main():
    args = build_parser().parse_args()
    np.random.seed(args.seed)

    if isinstance(args.hidden_size, list) and len(args.hidden_size) == 1:
        args.hidden_size = args.hidden_size[0]

    # ── Load config if available ──────────────────────────────────────────────
    cfg = vars(args).copy()
    if os.path.exists(args.config_path):
        with open(args.config_path) as f:
            saved = json.load(f)
        cfg.update(saved)
        print(f"Loaded config from {args.config_path}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading {args.dataset} …")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
        args.dataset, val_split=args.val_split, seed=args.seed
    )

    # ── Rebuild model ─────────────────────────────────────────────────────────
    model = NeuralNetwork(cfg)

    # ── Load weights ──────────────────────────────────────────────────────────
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    weights = np.load(args.model_path, allow_pickle=True).item()
    model.set_weights(weights)
    print(f"Loaded weights from {args.model_path}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    test_acc  = model.compute_accuracy(X_test, y_test)
    test_loss = model.compute_loss(X_test, y_test)
    y_pred    = model.predict(X_test)

    print(f"\nTest Accuracy : {test_acc:.4f}")
    print(f"Test Loss     : {test_loss:.4f}")

    try:
        from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall    = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1        = f1_score(y_test, y_pred, average="macro", zero_division=0)
        print(f"Precision     : {precision:.4f}")
        print(f"Recall        : {recall:.4f}")
        print(f"F1 (macro)    : {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                    target_names=get_class_names(args.dataset),
                                    zero_division=0))
    except Exception as err:
        print(f"[Warning] sklearn metrics failed: {err}")


if __name__ == "__main__":
    main()
