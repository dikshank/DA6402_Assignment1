
import numpy as np

class CrossEntropy:

    def softmax(self,x):
        exp = np.exp(x - np.max(x,axis=1,keepdims=True))
        return exp/np.sum(exp,axis=1,keepdims=True)

    def loss(self,y_true,logits):
        probs = self.softmax(logits)
        log_probs = -np.log(probs + 1e-9)
        return np.mean(np.sum(y_true*log_probs,axis=1))

    def grad(self,y_true,logits):
        probs = self.softmax(logits)
        return (probs - y_true)/y_true.shape[0]
