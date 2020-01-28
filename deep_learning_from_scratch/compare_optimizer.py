import matplotlib.pyplot as plt
from datasets.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import *


# 01 get data
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000

# optimizer
optimizer = {}
optimizer['SGD'] = SGD()
optimizer['Momentum'] = Momentum()
optimizer['AdaGrad'] = AdaGrad()
optimizer['Adam'] = Adam()

network = {}
train_loss = {}
for key in optimizer.keys():
    network[key] = MultiLayerNet(
        input_size=784, hidden_size_list=[100,100,100,100],
        output_size = 10)
    train_loss[key] = []

# train
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in optimizer.keys():
        grads = network[key].gradient(x_batch, t_batch)
        optimizer[key].update(network[key].params, grads)

        loss = network[key].loss(x_batch,t_batch)
        train_loss[key].append(loss)

    if i %100==0:
        print("========="+"iteration:"+str(i) + "=========")
        for key in optimizer.keys():
            loss = network[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))


# graph
markers = {"SGD":"o", "Momentum":"x", "AdaGrad":"s","Adam":"D"}
x=np.arange(max_iterations)
for key in optimizer.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0,1)
plt.legend()
plt.show()

