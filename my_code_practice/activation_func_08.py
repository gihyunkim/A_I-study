import numpy as np

class ReLu:
    def __init__(self):
        pass
    def forward(self,x):
        self.mask = (x<=0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self,dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class sigmoid:
    def __init__(self):
        self.out = None

    def forward(self,x):
        self.out = 1 / (1+np.exp(-x))
        return self.out

    def backward(self, dout):
        dx = dout * self.out * (1.0 - self.out)
        return dx