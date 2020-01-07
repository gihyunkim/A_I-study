import numpy as np
from functions import softmax, cross_entropy_error
from gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.w = np.random.randn(2,3)

    def predict(self,x):
        return np.dot(x, self.w)


    def loss(self,x, y):
        pred = self.predict(x)
        pred = softmax(pred)
        loss = cross_entropy_error(pred, y)
        return loss

# loss구하기
net = simpleNet()
w = net.w
x = np.array([0.6,0.9])
y = np.array([0,0,1])
pred = net.predict(x)
loss = net.loss(x,y)
print(loss)


def f(w):
    return net.loss(x,y)

dw = numerical_gradient(f, net.w)
print(dw)