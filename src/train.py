
import argparse
import numpy as np

from ann.neural_network import NeuralNetwork
from ann.neural_layer import NeuralLayer
from ann.activations import ReLU
from ann.objective_functions import CrossEntropy
from ann.optimizers import SGD
from utils.data_loader import load_dataset


def parse_arguments():

    parser=argparse.ArgumentParser()

    parser.add_argument("-d","--dataset",default="mnist")
    parser.add_argument("-e","--epochs",type=int,default=5)
    parser.add_argument("-b","--batch_size",type=int,default=32)
    parser.add_argument("-l","--loss",default="cross_entropy")
    parser.add_argument("-o","--optimizer",default="sgd")
    parser.add_argument("-lr","--learning_rate",type=float,default=0.001)
    parser.add_argument("-wd","--weight_decay",type=float,default=0)
    parser.add_argument("-nhl","--num_layers",type=int,default=2)
    parser.add_argument("-sz","--hidden_size",nargs="+",type=int,default=[128,128])
    parser.add_argument("-a","--activation",default="relu")
    parser.add_argument("-wi","--weight_init",default="random")
    parser.add_argument("-wp","--wandb_project",default="da6401")

    return parser.parse_args()


def train(args):

    X_train,y_train,X_test,y_test=load_dataset(args.dataset)

    model=NeuralNetwork(args)

    model.add(NeuralLayer(784,128,ReLU()))
    model.add(NeuralLayer(128,128,ReLU()))
    model.add(NeuralLayer(128,10))

    loss_fn=CrossEntropy()
    optimizer=SGD()

    n=len(X_train)

    for epoch in range(args.epochs):

        for i in range(0,n,args.batch_size):

            xb=X_train[i:i+args.batch_size]
            yb=y_train[i:i+args.batch_size]

            logits=model.forward(xb)

            loss=loss_fn.loss(yb,logits)

            grad=loss_fn.grad(yb,logits)

            model.backward(grad)

            optimizer.step(model.layers,args.learning_rate)

        print("epoch",epoch,"loss",loss)

    weights=model.get_weights()

    np.save("best_model.npy",weights)


if __name__=="__main__":

    args=parse_arguments()

    train(args)
