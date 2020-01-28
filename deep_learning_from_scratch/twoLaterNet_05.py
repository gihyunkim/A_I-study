from functions import *
from layers import *
from gradient import numerical_gradient
from collections import OrderedDict
import numpy as np

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std =0.01):
        self.params = {}
        self.params['w1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)



    def predict(self, x):
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x,w1) + b1
        z1 = sigmoid(a1)

        a2 = np.dot(z1,w2) + b2
        z2 = softmax(a2)
        return z2

    def loss(self,x, y):
        pred = self.predict(x)
        return cross_entropy_error(pred, y)

    def accuracy(self,x,y):
        pred = self.predict(x)
        max_x = np.argmax(pred, axis=1)
        max_y = np.argmax(y, axis=1)
        sum = np.sum(max_x == max_y) / float(x.shape[0])
        return sum

    def numerical_gradient(self,x, y):
        loss_w = lambda w: self.loss(x,y)
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']

        grad = {}
        grad['w1'] = numerical_gradient(loss_w, w1)
        grad['w2'] = numerical_gradient(loss_w, w2)
        grad['b1'] = numerical_gradient(loss_w, b1)
        grad['b2'] = numerical_gradient(loss_w, b2)

        return grad
