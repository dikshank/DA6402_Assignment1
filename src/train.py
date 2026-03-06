
import argparse
import numpy as np

from ann.neural_network import NeuralNetwork
from ann.objective_functions import CrossEntropy
from ann.optimizers import SGD
from utils.data_loader import load_dataset

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",default="mnist")
    parser.add_argument("--epochs",type=int,default=3)
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--learning_rate",type=float,default=0.01)
    parser.add_argument("--optimizer",default="sgd")
    parser.add_argument("--num_layers",type=int,default=2)
    parser.add_argument("--hidden_size",nargs="+",type=int,default=[128,128])
    parser.add_argument("--activation",default="relu")

    return parser.parse_args()

def train(args):

    X_train,y_train,X_test,y_test = load_dataset(args.dataset)

    model = NeuralNetwork(args)

    loss_fn = CrossEntropy()
    optimizer = SGD()

    n = len(X_train)

    for epoch in range(args.epochs):

        for i in range(0,n,args.batch_size):

            xb = X_train[i:i+args.batch_size]
            yb = y_train[i:i+args.batch_size]

            logits = model.forward(xb)

            loss = loss_fn.loss(yb,logits)

            grad = loss_fn.grad(yb,logits)

            model.backward(None,grad)

            optimizer.step(model.layers,args.learning_rate)

        print("epoch",epoch,"loss",loss)

    weights = model.get_weights()

    np.save("src/best_model.npy",weights)

if __name__ == "__main__":

    args = parse_arguments()

    train(args)
