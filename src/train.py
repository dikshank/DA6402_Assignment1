import argparse
import numpy as np

from ann.neural_network import NeuralNetwork
from ann.neural_layer import NeuralLayer
from ann.activations import ReLU
from ann.objective_functions import CrossEntropy
from ann.optimizers import SGD
from utils.data_loader import load_dataset

def train(args):
    X_train, y_train, X_test, y_test = load_dataset(args.dataset)

    model = NeuralNetwork()
    model.add(NeuralLayer(784, 128, ReLU()))
    model.add(NeuralLayer(128, 128, ReLU()))
    model.add(NeuralLayer(128, 10))

    loss_fn = CrossEntropy()
    optimizer = SGD()

    for epoch in range(args.epochs):
        logits = model.forward(X_train)
        loss = loss_fn.loss(y_train, logits)
        grad = loss_fn.grad(y_train, logits)
        model.backward(grad)
        optimizer.step(model.layers, args.lr)
        print("epoch", epoch, "loss", loss)

    weights = model.get_weights()
    np.save("best_model.npy", weights)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    train(args)
