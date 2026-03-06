
import numpy as np
from keras.datasets import mnist, fashion_mnist

def one_hot(y,num_classes=10):

    out = np.zeros((y.shape[0],num_classes))
    out[np.arange(y.shape[0]),y] = 1

    return out

def load_dataset(name="mnist"):

    if name == "mnist":
        (x_train,y_train),(x_test,y_test) = mnist.load_data()
    else:
        (x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()

    x_train = x_train.reshape(len(x_train),-1)/255.0
    x_test = x_test.reshape(len(x_test),-1)/255.0

    y_train = one_hot(y_train)
    y_test = one_hot(y_test)

    return x_train,y_train,x_test,y_test
