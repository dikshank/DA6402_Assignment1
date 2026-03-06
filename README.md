# DA6401 – Assignment 1: Multi-Layer Perceptron

> **NumPy-only MLP** for image classification on MNIST / Fashion-MNIST.

---

## Links

| Resource | URL |
|---|---|
| **GitHub Repository** | *(add your public repo link here)* |
| **W&B Report** | *(add your public W&B report link here)* |

---

## Project Structure

```
da6401_assignment1/
├── README.md
├── requirements.txt
├── models/                    # saved .npy weight files
├── notebooks/
│   └── wandb_demo.ipynb       # W&B experiment notebook
└── src/
    ├── train.py               # training entry point
    ├── inference.py           # evaluation entry point
    ├── best_model.npy         # best model weights (by test F1)
    ├── best_config.json       # best hyperparameter config
    ├── ann/
    │   ├── activations.py     # sigmoid, tanh, relu, softmax
    │   ├── neural_layer.py    # single dense layer (forward + backward)
    │   ├── neural_network.py  # full MLP (forward, backward, fit, get/set weights)
    │   ├── objective_functions.py  # cross-entropy, MSE + gradients
    │   ├── optimizers.py      # SGD, Momentum, NAG, RMSProp
    │   └── __init__.py
    └── utils/
        ├── data_loader.py     # MNIST / Fashion-MNIST loader
        └── __init__.py
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Training

```bash
# Default (Fashion-MNIST, RMSProp, ReLU, Xavier, cross-entropy)
python src/train.py

# Custom configuration
python src/train.py \
  -d fashion_mnist \
  -e 20 \
  -b 32 \
  -o rmsprop \
  -lr 0.001 \
  -wd 0.0 \
  -nhl 3 \
  -sz 128 \
  -a relu \
  -l cross_entropy \
  -wi xavier \
  -wp da6401_assignment1
```

### CLI Arguments

| Flag | Long form | Default | Description |
|------|-----------|---------|-------------|
| `-d` | `--dataset` | `fashion_mnist` | `mnist` or `fashion_mnist` |
| `-e` | `--epochs` | `10` | Training epochs |
| `-b` | `--batch_size` | `32` | Mini-batch size |
| `-l` | `--loss` | `cross_entropy` | `cross_entropy` or `mean_squared_error` |
| `-o` | `--optimizer` | `rmsprop` | `sgd`, `momentum`, `nag`, `rmsprop` |
| `-lr`| `--learning_rate` | `0.001` | Initial learning rate |
| `-wd`| `--weight_decay` | `0.0` | L2 regularisation coefficient |
| `-nhl`| `--num_layers` | `3` | Number of **hidden** layers |
| `-sz`| `--hidden_size` | `128` | Neurons per hidden layer |
| `-a` | `--activation` | `relu` | `sigmoid`, `tanh`, `relu` |
| `-wi`| `--weight_init` | `xavier` | `random` or `xavier` |
| `-wp`| `--wandb_project` | `da6401_assignment1` | W&B project name |

---

## Inference

```bash
python src/inference.py \
  --model_path src/best_model.npy \
  -d fashion_mnist
```

Outputs: **Accuracy**, **Precision**, **Recall**, **F1-score** (macro).

---

## Implementation Notes

- **Pure NumPy**: no PyTorch, TensorFlow, or autograd.
- `forward()` returns **raw logits** (no softmax applied).
- `backward()` computes gradients from **last layer to first**; each layer exposes `self.grad_W` and `self.grad_b`.
- Softmax is fused with the loss functions for numerical stability.
- Model weights are saved/loaded via `get_weights()` / `set_weights()`.
