import numpy as np
import pickle
import sys, os
sys.path.append(os.pardir)
from datasets.mnist import load_mnist
from perceptron_01 import sigmoid, softmax

def get_data():
    (x_train, y_train), (x_test, y_test) = load_mnist(flatten=True, one_hot_label=False, normalize=True)
    return x_test, y_test

def init_network():
    with open("sample_weight.pkl",'rb') as f:
        network = pickle.load(f)
    return network

def predict(network ,x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    z3 = softmax(a3)
    return z3

x, y = get_data()
network = init_network()
cnt = 0
for i in range(len(x)):
    pred = predict(network, x[i])
    p = np.argmax(pred)

    if p == y[i]:
        cnt+=1

print(cnt / len(x) * 100)

