"""
train.py – Train the MLP on MNIST or Fashion-MNIST.

Usage:
    python src/train.py -d mnist -e 10 -b 32 -o adam -lr 0.001 \
                        -nhl 3 -sz 128 -a relu -l cross_entropy -wi xavier
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
from utils.data_loader import load_dataset


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_arguments():
    p = argparse.ArgumentParser(description="Train MLP on MNIST / Fashion-MNIST")

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
    p.add_argument("--model_path",             type=str,   default="best_model.npy")
    return p.parse_args()


# Alias
parse_args = parse_arguments


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_arguments()
    np.random.seed(args.seed)

    # Flatten hidden_size: argparse nargs="+" always gives a list
    if isinstance(args.hidden_size, list):
        if len(args.hidden_size) == 1:
            args.hidden_size = args.hidden_size[0]
        # multi-element list stays as-is; NeuralNetwork handles it via hidden_sizes

    # ── W&B setup ────────────────────────────────────────────────────────────
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config=vars(args),
            )
        except ImportError:
            print("[Warning] wandb not installed. Continuing without logging.")
        except Exception as e:
            print(f"[Warning] W&B init failed: {e}. Continuing without logging.")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading {args.dataset} …")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
        args.dataset, val_split=args.val_split, seed=args.seed
    )

    # ── Build & train ─────────────────────────────────────────────────────────
    model = NeuralNetwork(args)
    best_weights, best_val_acc = model.fit(
        X_train, y_train, X_val, y_val, args, wandb_run=wandb_run
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    model.set_weights(best_weights)
    test_acc  = model.compute_accuracy(X_test, y_test)
    test_loss = model.compute_loss(X_test, y_test)
    print(f"\nBest Val Acc : {best_val_acc:.4f}")
    print(f"Test Acc     : {test_acc:.4f}  |  Test Loss: {test_loss:.4f}")

    f1 = 0.0
    try:
        from sklearn.metrics import f1_score, classification_report
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro")
        print(f"Test F1 (macro): {f1:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))
    except Exception:
        pass

    # ── Save model & config ────────────────────────────────────────────────────
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    model_path  = os.path.join(save_dir, "best_model.npy")
    config_path = os.path.join(save_dir, "best_config.json")

    np.save(model_path, best_weights)

    # Also save JSON weights (more reliable on autograder)
    json_weights_path = os.path.join(save_dir, "best_model.json")
    with open(json_weights_path, 'w') as f:
        json.dump({k: v.tolist() for k, v in best_weights.items()}, f)

    hidden = args.hidden_size if isinstance(args.hidden_size, list) \
             else [args.hidden_size] * args.num_layers

    config = {
        "dataset":       args.dataset,
        "num_layers":    args.num_layers,
        "hidden_size":   hidden,
        "hidden_sizes":  hidden,
        "activation":    args.activation,
        "optimizer":     args.optimizer,
        "learning_rate": args.learning_rate,
        "weight_decay":  args.weight_decay,
        "loss":          args.loss,
        "weight_init":   args.weight_init,
        "batch_size":    args.batch_size,
        "epochs":        args.epochs,
        "val_split":     args.val_split,
        "seed":          args.seed,
        "test_accuracy": float(test_acc),
        "test_f1":       float(f1),
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved model  → {model_path}")
    print(f"Saved config → {config_path}")

    if wandb_run is not None:
        wandb_run.summary["best_val_acc"] = best_val_acc
        wandb_run.summary["test_acc"]     = test_acc
        wandb_run.summary["test_f1"]      = f1
        wandb_run.finish()


if __name__ == "__main__":
    main()
