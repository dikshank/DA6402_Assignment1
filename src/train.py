"""
train.py – Train the MLP on MNIST or Fashion-MNIST.

Usage example:
    python train.py -d mnist -e 10 -b 32 -o rmsprop -lr 0.001 \
                    -nhl 3 -sz 128 -a relu -l cross_entropy \
                    -wi xavier -wd 0.0
"""
import os
import sys
import json
import argparse
import numpy as np

# Allow imports from the src directory
sys.path.insert(0, os.path.dirname(__file__))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="Train MLP on MNIST / Fashion-MNIST")

    p.add_argument("-d",   "--dataset",       type=str,   default="fashion_mnist",
                   choices=["mnist", "fashion_mnist"],
                   help="Dataset to train on")
    p.add_argument("-e",   "--epochs",         type=int,   default=10,
                   help="Number of training epochs")
    p.add_argument("-b",   "--batch_size",     type=int,   default=32,
                   help="Mini-batch size")
    p.add_argument("-l",   "--loss",           type=str,   default="cross_entropy",
                   choices=["cross_entropy", "mean_squared_error"],
                   help="Loss function")
    p.add_argument("-o",   "--optimizer",      type=str,   default="rmsprop",
                   choices=["sgd", "momentum", "nag", "rmsprop"],
                   help="Optimizer")
    p.add_argument("-lr",  "--learning_rate",  type=float, default=0.001,
                   help="Initial learning rate")
    p.add_argument("-wd",  "--weight_decay",   type=float, default=0.0,
                   help="L2 weight decay coefficient")
    p.add_argument("-nhl", "--num_layers",     type=int,   default=3,
                   help="Number of hidden layers")
    p.add_argument("-sz",  "--hidden_size",    type=int,   default=128,
                   nargs="+",
                   help="Neurons per hidden layer (single int or space-separated list)")
    p.add_argument("-a",   "--activation",     type=str,   default="relu",
                   choices=["sigmoid", "tanh", "relu"],
                   help="Hidden-layer activation function")
    p.add_argument("-wi",  "--weight_init",    type=str,   default="xavier",
                   choices=["random", "xavier"],
                   help="Weight initialisation strategy")
    p.add_argument("-wp",  "--wandb_project",  type=str,   default="da6401_assignment1",
                   help="Weights & Biases project name")
    p.add_argument("--wandb_entity",           type=str,   default=None,
                   help="Weights & Biases entity (username / team)")
    p.add_argument("--no_wandb",               action="store_true",
                   help="Disable W&B logging")
    p.add_argument("--val_split",              type=float, default=0.1,
                   help="Fraction of training data used for validation")
    p.add_argument("--seed",                   type=int,   default=42,
                   help="Random seed")
    p.add_argument("--save_dir",               type=str,   default=".",
                   help="Directory to save best_model.npy and best_config.json")
    # Kept for inference.py parity
    p.add_argument("--model_path",             type=str,   default="best_model.npy",
                   help="(inference only) Path to serialised model weights")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args   = parser.parse_args()

    np.random.seed(args.seed)

    # ---- Normalise hidden_size ----------------------------------------
    if isinstance(args.hidden_size, list) and len(args.hidden_size) == 1:
        args.hidden_size = args.hidden_size[0]

    # ---- W&B setup ----------------------------------------------------
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

    # ---- Load data ----------------------------------------------------
    print(f"Loading {args.dataset} …")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
        args.dataset, val_split=args.val_split, seed=args.seed
    )
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # ---- Build model --------------------------------------------------
    model = NeuralNetwork(args)

    # ---- Train --------------------------------------------------------
    best_weights, best_val_acc = model.fit(
        X_train, y_train, X_val, y_val, args, wandb_run=wandb_run
    )

    # ---- Evaluate on test set ----------------------------------------
    model.set_weights(best_weights)
    test_acc  = model.compute_accuracy(X_test, y_test)
    test_loss = model.compute_loss(X_test, y_test)
    print(f"\nBest Val Acc : {best_val_acc:.4f}")
    print(f"Test Acc     : {test_acc:.4f}  |  Test Loss: {test_loss:.4f}")

    # ---- Compute F1 (used to pick best model) -----------------------
    from sklearn.metrics import f1_score, classification_report
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Test F1 (macro): {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ---- Save best model & config ------------------------------------
    os.makedirs(args.save_dir, exist_ok=True)
    model_path  = os.path.join(args.save_dir, "best_model.npy")
    config_path = os.path.join(args.save_dir, "best_config.json")

    np.save(model_path, best_weights)
    config = {
        "dataset":       args.dataset,
        "num_layers":    args.num_layers,
        "hidden_size":   args.hidden_size if isinstance(args.hidden_size, list)
                         else [args.hidden_size] * args.num_layers,
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

    # ---- Final W&B logging -------------------------------------------
    if wandb_run is not None:
        wandb_run.summary["best_val_acc"] = best_val_acc
        wandb_run.summary["test_acc"]     = test_acc
        wandb_run.summary["test_f1"]      = f1
        wandb_run.finish()


if __name__ == "__main__":
    main()
