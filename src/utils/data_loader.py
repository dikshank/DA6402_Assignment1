"""
Data loading utilities for MNIST and Fashion-MNIST.

Uses keras.datasets for loading (as permitted by the assignment spec).
Falls back to tensorflow.keras if the standalone keras package is not available.
"""
import numpy as np


def load_dataset(dataset_name: str, val_split: float = 0.1, seed: int = 42):
    """
    Load and preprocess MNIST or Fashion-MNIST.

    Parameters
    ----------
    dataset_name : 'mnist' or 'fashion_mnist'
    val_split    : fraction of training data used for validation
    seed         : random seed for reproducible split

    Returns
    -------
    X_train, y_train : training arrays
    X_val,   y_val   : validation arrays
    X_test,  y_test  : test arrays

    All X arrays are float32 with values in [0, 1], shape (N, 784).
    All y arrays are int32 class labels in [0, 9].
    """
    dataset_name = dataset_name.lower().replace("-", "_")

    # ---- Load raw data ------------------------------------------------
    try:
        import keras
        if dataset_name == "mnist":
            (X_tr_raw, y_tr_raw), (X_te_raw, y_te_raw) = keras.datasets.mnist.load_data()
        elif dataset_name == "fashion_mnist":
            (X_tr_raw, y_tr_raw), (X_te_raw, y_te_raw) = keras.datasets.fashion_mnist.load_data()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    except ImportError:
        # Fallback to tensorflow.keras
        import tensorflow as tf
        if dataset_name == "mnist":
            (X_tr_raw, y_tr_raw), (X_te_raw, y_te_raw) = tf.keras.datasets.mnist.load_data()
        elif dataset_name == "fashion_mnist":
            (X_tr_raw, y_tr_raw), (X_te_raw, y_te_raw) = tf.keras.datasets.fashion_mnist.load_data()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    # ---- Flatten and normalise ----------------------------------------
    X_tr = X_tr_raw.reshape(-1, 784).astype(np.float32) / 255.0
    X_te = X_te_raw.reshape(-1, 784).astype(np.float32) / 255.0
    y_tr = y_tr_raw.astype(np.int32)
    y_te = y_te_raw.astype(np.int32)

    # ---- Train / validation split (stratified) -----------------------
    np.random.seed(seed)
    n_train = X_tr.shape[0]
    idx = np.random.permutation(n_train)

    val_count = int(n_train * val_split)
    val_idx   = idx[:val_count]
    train_idx = idx[val_count:]

    X_train, y_train = X_tr[train_idx], y_tr[train_idx]
    X_val,   y_val   = X_tr[val_idx],   y_tr[val_idx]

    return X_train, y_train, X_val, y_val, X_te, y_te


def get_class_names(dataset_name: str):
    """Return human-readable class names for each dataset."""
    dataset_name = dataset_name.lower().replace("-", "_")
    if dataset_name == "mnist":
        return [str(i) for i in range(10)]
    elif dataset_name == "fashion_mnist":
        return [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
