import numpy as np
import torch.nn as nn

torch_dropout = nn.Dropout()
class Dropout():
    def __init__(self, p=0.5):
        # p is the probability of zeros
        self.p = p
        self.mask = None
    
    def forward(self, X, training=True):
        if training:
            # the E is 1-self.p so divide 1-self.p
            self.mask = np.random.binomial(1, 1 - self.p, size=X.shape) / (1 - self.p)
            return X * self.mask
        else:
            return X
    
    def backward(self, dY):
        return dY * self.mask

input = np.random.randn(5,4)
b = np.random.binomial(1, 0.1, size=(5,4))

print(b)
# my_dropout = Dropout(p=0.2)
# input = my_dropout.forward(input)
# print(input)