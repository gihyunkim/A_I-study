import numpy as np


# AND Module
def and_gate(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7

    if (np.sum(x * w) + b < 0):
        return 0
    else:
        return 1

# OR Module
def or_gate(x1, x2):
        x = np.array([x1, x2])
        w = np.array([0.5, 0.5])
        b = -0.2
        tmp = np.sum(x * w) + b

        if tmp < 0:
            return 0
        else:
            return 1

# step_function
def step_function(x):
    y = x>=0
    return y.astype(np.int)

# sigmoid function
def sigmoid(x):
    return 1 / (1+ np.exp(-x))

def relu(x):
    return np.maximum(0,x)


# 기본 신경망
def identity_func(x):
    return x


def init_network():
    network = {}
    network['w1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])

    network['w2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])

    network['w3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network


def forward(network, x):
    w1, w2, w3 = network['w1'], network['w2'], network['w3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # input -> hidden 1
    a1 = x.dot(w1) + b1
    z1 = sigmoid(a1)

    # hidden 1 -> hidden 2
    a2 = z1.dot(w2) + b2
    z2 = sigmoid(a2)

    # hidden 2 -> output
    a3 = z2.dot(w3) + b3
    o = identity_func(a3)
    return o


#출력층
# softmax
def softmax(x):
    max_x = np.max(x)

    exp_x = np.exp(x - max_x)

    sum_exp = np.sum(exp_x)

    result = exp_x / sum_exp

    return result