import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset

def evaluate(model, X, y):
    logits = model.forward(X)
    preds = np.argmax(logits, axis=1)
    y_true = np.argmax(y, axis=1)

    print("Accuracy", accuracy_score(y_true, preds))
    print("Precision", precision_score(y_true, preds, average="macro"))
    print("Recall", recall_score(y_true, preds, average="macro"))
    print("F1", f1_score(y_true, preds, average="macro"))
