"""
Data loading for MNIST and Fashion-MNIST.

Load priority (fastest/most reliable on Gradescope autograder first):
1. standalone keras datasets  (pre-installed, no network needed)
2. tensorflow.keras
3. direct download (network fallback)
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split


def _load_raw(name):
    """Try all sources to get (x_train, y_train), (x_test, y_test)."""

    # 1) Standalone keras (pre-installed on Gradescope, fastest)
    try:
        if name == "mnist":
            from keras.datasets import mnist
            return mnist.load_data()
        else:
            from keras.datasets import fashion_mnist
            return fashion_mnist.load_data()
    except Exception:
        pass

    # 2) tensorflow.keras
    try:
        import tensorflow as tf
        if name == "mnist":
            return tf.keras.datasets.mnist.load_data()
        else:
            return tf.keras.datasets.fashion_mnist.load_data()
    except Exception:
        pass

    # 3) Direct download (needs network)
    try:
        return _download_and_parse(name)
    except Exception:
        pass

    raise RuntimeError(f"Cannot load dataset '{name}' from any source.")


def _download_and_parse(name):
    import struct, gzip, urllib.request
    base = ("https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
            if name == "fashion_mnist"
            else "https://storage.googleapis.com/cvdf-datasets/mnist/")
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images":  "t10k-images-idx3-ubyte.gz",
        "test_labels":  "t10k-labels-idx1-ubyte.gz",
    }
    cache = os.path.join(os.path.expanduser("~"), ".datasets", name)
    os.makedirs(cache, exist_ok=True)
    data = {}
    for key, fname in files.items():
        fpath = os.path.join(cache, fname)
        if not os.path.exists(fpath):
            urllib.request.urlretrieve(base + fname, fpath)
        with gzip.open(fpath, "rb") as f:
            raw = f.read()
        if "images" in key:
            _, n, h, w = struct.unpack(">IIII", raw[:16])
            data[key] = np.frombuffer(raw[16:], np.uint8).reshape(n, h, w)
        else:
            _, n = struct.unpack(">II", raw[:8])
            data[key] = np.frombuffer(raw[8:], np.uint8)
    return ((data["train_images"], data["train_labels"]),
            (data["test_images"],  data["test_labels"]))


def load_dataset(dataset_name: str, val_split: float = 0.1, seed: int = 42):
    """
    Returns: X_train, y_train, X_val, y_val, X_test, y_test
    X: float32 in [0,1], shape (N, 784).  y: int32 in [0,9].
    """
    name = dataset_name.lower().replace("-", "_")
    (x_tr, y_tr), (x_te, y_te) = _load_raw(name)

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


# Alias used by friend's code style
def load_data(dataset_name: str, val_split: float = 0.1, seed: int = 42):
    return load_dataset(dataset_name, val_split, seed)


def get_class_names(dataset_name: str):
    name = dataset_name.lower().replace("-", "_")
    if name == "mnist":
        return [str(i) for i in range(10)]
    elif name == "fashion_mnist":
        return ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    raise ValueError(f"Unknown dataset: {dataset_name}")
