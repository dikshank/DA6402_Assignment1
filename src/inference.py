
import argparse
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

from ann.neural_network import NeuralNetwork
from ann.neural_layer import NeuralLayer
from ann.activations import ReLU
from utils.data_loader import load_dataset


def parse_arguments():

    parser=argparse.ArgumentParser()

    parser.add_argument("-d","--dataset",default="mnist")
    parser.add_argument("--model_path",default="best_model.npy")

    return parser.parse_args()


def load_model(path):
    return np.load(path,allow_pickle=True)


def evaluate():

    args=parse_arguments()

    X_train,y_train,X_test,y_test=load_dataset(args.dataset)

    model=NeuralNetwork()

    model.add(NeuralLayer(784,128,ReLU()))
    model.add(NeuralLayer(128,128,ReLU()))
    model.add(NeuralLayer(128,10))

    weights=load_model(args.model_path)

    model.set_weights(weights)

    logits=model.forward(X_test)

    preds=np.argmax(logits,axis=1)
    y_true=np.argmax(y_test,axis=1)

    print("Accuracy",accuracy_score(y_true,preds))
    print("Precision",precision_score(y_true,preds,average="macro"))
    print("Recall",recall_score(y_true,preds,average="macro"))
    print("F1",f1_score(y_true,preds,average="macro"))


if __name__=="__main__":
    evaluate()
