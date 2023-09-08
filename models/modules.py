import torch
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

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        output = (x - mean) / (std + self.eps)
        output = self.weight * output + self.bias
        self.mean = mean
        self.std = std
        return output

    def backward(self, grad_output):
        x = self.input
        mean = self.mean
        std = self.std
        grad_weight = (grad_output * ((x - mean) / (std + self.eps))).sum(-1)
        grad_bias = grad_output.sum(-1)
        grad_x = grad_output * self.weight / (std + self.eps)
        grad_x -= ((grad_output * self.weight).sum(-1, keepdim=True) * (x - mean) * (std + self.eps) ** -2) / x.size(-1)
        return grad_x, grad_weight, grad_bias

    def __call__(self, x):
        self.input = x
        output = self.forward(x)
        return output

input = np.random.randn(5,4)
b = np.random.binomial(1, 0.1, size=(5,4))

print(b)
# my_dropout = Dropout(p=0.2)
# input = my_dropout.forward(input)
# print(input)