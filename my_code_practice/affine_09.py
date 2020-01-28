import numpy as np

class affine:
    def __init__(self,w,b):
        self.w = w
        self.b = b
        self.x = None
        self.dw = None

    def forward(self,x):
        self.x = x
        return np.dot(self.x, self.w)+self.b

    def backward(self, dout):
        self.dw = np.dot(self.x.T,dout)
        dx = np.dot(dout ,self.w.T)
        self.db = np.sum(dout,axis=0)
        return dx


from functions import softmax, cross_entropy_error
class softmax_with_loss:
    def __init__(self):
        self.loss =None
        self.x = None
        self.y = None

    def forward(self,x, y):
        self.x = softmax(x)
        self.y = y
        self.loss = cross_entropy_error(self.x, self.y)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.x.shape[0]
        dx =  (self.x-self.y) / batch_size
        return dx


