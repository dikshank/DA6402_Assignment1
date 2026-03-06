import numpy as np

class NeuralLayer:
    def __init__(self, in_features, out_features, activation=None):
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros((1, out_features))
        self.activation = activation

    def forward(self, X):
        self.X = X
        self.Z = X @ self.W + self.b
        if self.activation:
            return self.activation.forward(self.Z)
        return self.Z

    def backward(self, grad):
        if self.activation:
            grad = self.activation.backward(grad)
        self.grad_W = self.X.T @ grad
        self.grad_b = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.W.T
