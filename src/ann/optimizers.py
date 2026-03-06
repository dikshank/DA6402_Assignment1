
class SGD:
    def step(self,layers,lr):
        for layer in layers:
            layer.W-=lr*layer.grad_W
            layer.b-=lr*layer.grad_b
