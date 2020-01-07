import numpy as np
from datasets.mnist import load_mnist
from back_prop_two_layerNet import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size = 50, output_size = 10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_backprop = network.gradient(x_batch, t_batch)


