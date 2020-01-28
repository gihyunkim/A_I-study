from datasets.mnist import load_mnist
import matplotlib.pyplot as plt
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD
import numpy as np

# Data
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000

# network
weight_init_type = {"std=0.01":0.01, "xavier":"sigmoid", "he":"relu"}

network = {}
optimizer = SGD(lr=0.01)
train_loss = {}
for key, val in weight_init_type.items():
    network[key] = MultiLayerNet(input_size=784, hidden_size_list=[100,100,100,100], output_size=10,
                                 weight_init_std=val)
    train_loss[key]=[]

for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in weight_init_type.keys():
        grads = network[key].gradient(x_batch, t_batch)
        optimizer.update(network[key].params, grads)
        loss = network[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)


    if i%100==0:
        print("======="+str(i)+"=======")
        for key in weight_init_type.keys():
            loss = network[key].loss(x_batch, t_batch)
            print(key+" activation loss:"+str(loss))


markers = {"std=0.01":"o", "xavier":"s", "he":"D"}
x = np.arange(max_iterations)

for key in weight_init_type.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker = markers[key], label=key, markevery=100)
plt.ylabel("loss")
plt.xlabel("iterations")
plt.ylim(0,2.5)
plt.legend()
plt.show()