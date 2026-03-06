"""
inference.py – Load a serialised MLP and evaluate it.

Usage:
    python inference.py --model_path best_model.npy -d mnist
    python inference.py --model_path best_model.npy -d fashion_mnist
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


# ---------------------------------------------------------------------------
# CLI  (kept identical to train.py as required by updated instructions)
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="Inference with a saved MLP model")

    p.add_argument("-d",   "--dataset",       type=str,   default="fashion_mnist",
                   choices=["mnist", "fashion_mnist"])
    p.add_argument("-e",   "--epochs",         type=int,   default=10)
    p.add_argument("-b",   "--batch_size",     type=int,   default=32)
    p.add_argument("-l",   "--loss",           type=str,   default="cross_entropy",
                   choices=["cross_entropy", "mean_squared_error"])
    p.add_argument("-o",   "--optimizer",      type=str,   default="rmsprop",
                   choices=["sgd", "momentum", "nag", "rmsprop"])
    p.add_argument("-lr",  "--learning_rate",  type=float, default=0.001)
    p.add_argument("-wd",  "--weight_decay",   type=float, default=0.0)
    p.add_argument("-nhl", "--num_layers",     type=int,   default=3)
    p.add_argument("-sz",  "--hidden_size",    type=int,   default=128, nargs="+")
    p.add_argument("-a",   "--activation",     type=str,   default="relu",
                   choices=["sigmoid", "tanh", "relu"])
    p.add_argument("-wi",  "--weight_init",    type=str,   default="xavier",
                   choices=["random", "xavier"])
    p.add_argument("-wp",  "--wandb_project",  type=str,   default="da6401_assignment1")
    p.add_argument("--wandb_entity",           type=str,   default=None)
    p.add_argument("--no_wandb",               action="store_true")
    p.add_argument("--val_split",              type=float, default=0.1)
    p.add_argument("--seed",                   type=int,   default=42)
    p.add_argument("--save_dir",               type=str,   default=".")
    p.add_argument("--model_path",             type=str,   default="best_model.npy",
                   help="Path to saved .npy model weights")

    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(model_path):
    """Load trained model weights from disk."""
    data = np.load(model_path, allow_pickle=True).item()
    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args   = parser.parse_args()

    # ---- Normalise hidden_size ----------------------------------------
    if isinstance(args.hidden_size, list) and len(args.hidden_size) == 1:
        args.hidden_size = args.hidden_size[0]

    # ---- Try to load config from best_config.json if available --------
    config_path = os.path.join(os.path.dirname(args.model_path), "best_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            saved_cfg = json.load(f)
        # Override relevant args from saved config
        for key in ("num_layers", "hidden_size", "activation", "weight_init",
                    "loss", "dataset"):
            if key in saved_cfg:
                setattr(args, key, saved_cfg[key])
        print(f"Loaded config from {config_path}")

    # ---- Load data ----------------------------------------------------
    print(f"Loading {args.dataset} …")
    _, _, _, _, X_test, y_test = load_dataset(
        args.dataset, val_split=args.val_split, seed=args.seed
    )

    # ---- Build and load model ----------------------------------------
    model = NeuralNetwork(args)
    weights = load_model(args.model_path)
    model.set_weights(weights)
    print(f"Loaded weights from {args.model_path}")

    # ---- Inference ----------------------------------------------------
    y_pred = model.predict(X_test)

    # ---- Metrics ------------------------------------------------------
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, classification_report, confusion_matrix
    )

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall    = recall_score(y_test, y_pred,    average="macro", zero_division=0)
    f1        = f1_score(y_test, y_pred,        average="macro", zero_division=0)

    print("\n" + "=" * 50)
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}  (macro)")
    print(f"  Recall    : {recall:.4f}  (macro)")
    print(f"  F1-score  : {f1:.4f}  (macro)")
    print("=" * 50)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
    }


if __name__ == "__main__":
    main()
