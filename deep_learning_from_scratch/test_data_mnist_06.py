import os, sys
import numpy as np
sys.path.append(os.pardir) # 부모 디렉터리 참조 가능

from datasets.mnist import load_mnist
from twoLaterNet_05 import TwoLayerNet

(x_train, y_train), (x_test, y_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size= 784, hidden_size=50, output_size=10)
iters_num = 10000
train_size =  x_train.shape[0]
batch_size = 100
learning_rate= 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(iters_num / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]

    # gradient
    grad = network.numerical_gradient(x_batch, y_batch)

    for key in ('W1', 'W2', 'b1','b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, y_batch)
    train_loss_list.append(loss)

    # 1 epcoh당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_batch, y_batch)
        test_acc  = network.accuracy(x_test, y_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train_acc: %f, test_acc: %f"%(train_acc,test_acc))
