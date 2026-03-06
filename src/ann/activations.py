
import numpy as np

class Sigmoid:
    def forward(self, x):
        self.out = 1/(1+np.exp(-x))
        return self.out
    def backward(self, grad):
        return grad * self.out * (1-self.out)

class Tanh:
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out
    def backward(self, grad):
        return grad * (1-self.out**2)

class ReLU:
    def forward(self, x):
        self.mask = x>0
        return x*self.mask
    def backward(self, grad):
        return grad*self.mask
