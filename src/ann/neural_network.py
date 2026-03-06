class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def get_weights(self):
        weights = []
        for layer in self.layers:
            weights.append({"W": layer.W, "b": layer.b})
        return weights

    def set_weights(self, weights):
        for layer, w in zip(self.layers, weights):
            layer.W = w["W"]
            layer.b = w["b"]
