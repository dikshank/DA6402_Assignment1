"""
Data loading for MNIST and Fashion-MNIST.

Load priority:
1. Direct download (pure Python / NumPy – no ML framework needed)
2. standalone keras datasets
3. tensorflow.keras
"""
import os
import struct
import gzip
import urllib.request
import numpy as np
from sklearn.model_selection import train_test_split


# ── Direct download (always works, no framework dependency) ─────────────────

_MNIST_BASE   = "https://storage.googleapis.com/cvdf-datasets/mnist/"
_FMNIST_BASE  = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"

_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}


def _download_and_parse(dataset_name: str):
    base = _FMNIST_BASE if dataset_name == "fashion_mnist" else _MNIST_BASE
    cache = os.path.join(os.path.expanduser("~"), ".datasets", dataset_name)
    os.makedirs(cache, exist_ok=True)
    data = {}
    for key, fname in _FILES.items():
        fpath = os.path.join(cache, fname)
        if not os.path.exists(fpath):
            print(f"  Downloading {fname} ...")
            urllib.request.urlretrieve(base + fname, fpath)
        with gzip.open(fpath, "rb") as f:
            raw = f.read()
        if "images" in key:
            _, n, h, w = struct.unpack(">IIII", raw[:16])
            data[key] = np.frombuffer(raw[16:], np.uint8).reshape(n, h, w)
        else:
            _, n = struct.unpack(">II", raw[:8])
            data[key] = np.frombuffer(raw[8:], np.uint8)
    return (data["train_images"], data["train_labels"]), \
           (data["test_images"],  data["test_labels"])


# ── Public API ───────────────────────────────────────────────────────────────

def load_dataset(dataset_name: str, val_split: float = 0.1, seed: int = 42):
    """
    Load and preprocess MNIST or Fashion-MNIST.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test
    X arrays: float32 in [0,1], shape (N, 784).
    y arrays: int32 in [0,9].
    """
    name = dataset_name.lower().replace("-", "_")
    x_tr = y_tr = x_te = y_te = None

    # 1) Direct download (most reliable on autograder)
    try:
        (x_tr, y_tr), (x_te, y_te) = _download_and_parse(name)
    except Exception:
        pass

    # 2) Standalone keras
    if x_tr is None:
        try:
            if name == "mnist":
                from keras.datasets import mnist
                (x_tr, y_tr), (x_te, y_te) = mnist.load_data()
            else:
                from keras.datasets import fashion_mnist
                (x_tr, y_tr), (x_te, y_te) = fashion_mnist.load_data()
        except Exception:
            pass

    # 3) tensorflow.keras
    if x_tr is None:
        try:
            import tensorflow as tf
            if name == "mnist":
                (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
            else:
                (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.fashion_mnist.load_data()
        except Exception:
            pass

    if x_tr is None:
        raise RuntimeError(f"Could not load dataset '{dataset_name}'.")

    X_tr = x_tr.reshape(-1, 784).astype(np.float32) / 255.0
    X_te = x_te.reshape(-1, 784).astype(np.float32) / 255.0
    y_tr = y_tr.astype(np.int32)
    y_te = y_te.astype(np.int32)

    X_train, X_val, y_train, y_val = train_test_split(
        X_tr, y_tr, test_size=val_split, random_state=seed, stratify=y_tr
    )

    print(f"Dataset: {dataset_name} | Train: {X_train.shape[0]} | "
          f"Val: {X_val.shape[0]} | Test: {X_te.shape[0]}")

    return X_train, y_train, X_val, y_val, X_te, y_te


def get_class_names(dataset_name: str):
    name = dataset_name.lower().replace("-", "_")
    if name == "mnist":
        return [str(i) for i in range(10)]
    elif name == "fashion_mnist":
        return ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    raise ValueError(f"Unknown dataset: {dataset_name}")
