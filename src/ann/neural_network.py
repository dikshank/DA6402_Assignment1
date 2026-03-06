from .neural_layer import NeuralLayer
from .activations import ReLU, Sigmoid, Tanh


class NeuralNetwork:

    def __init__(self, args):

        self.layers = []

        input_size = 784

        # activation selection
        if args.activation == "relu":
            activation = ReLU
        elif args.activation == "sigmoid":
            activation = Sigmoid
        else:
            activation = Tanh

        # hidden layers
        for size in args.hidden_size:
            self.layers.append(
                NeuralLayer(input_size, size, activation())
            )
            input_size = size

        # output layer (logits)
        self.layers.append(
            NeuralLayer(input_size, 10)
        )


    # forward pass
    def forward(self, X):

        out = X

        for layer in self.layers:
            out = layer.forward(out)

        return out


    # backward pass
    def backward(self, X, grad):

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return grad


    # return weights for saving
    def get_weights(self):

        weights = []

        for layer in self.layers:
            weights.append({
                "W": layer.W,
                "b": layer.b
            })

        return weights


    # load weights (used by autograder)
    def set_weights(self, weights):

        for layer, w in zip(self.layers, weights):

            layer.W = w["W"]
            layer.b = w["b"]