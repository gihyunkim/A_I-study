from datasets.mnist import load_mnist
from common.util import shuffle_dataset
from common.multi_layer_net import MultiLayerNet
from common.trainer import Trainer
import numpy as np
import matplotlib.pyplot as plt

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

x_train = x_train[:500]
t_train = t_train[:500]

validation_ratio = 0.2
train_size = x_train.shape[0]
val_size = int(train_size * validation_ratio)

x_train, t_train = shuffle_dataset(x_train, t_train)

x_val = x_train[:val_size]
t_val = t_train[:val_size]
x_train = x_train[val_size:]
t_train = t_train[val_size:]

def __train(lr, weight_decay, epochs=50):
    network = MultiLayerNet(input_size=784, hidden_size_list=[100,100,100,100,100,100]
                            ,output_size=10, weight_decay_lambda=weight_decay)
    trainer = Trainer(network, x_train, t_train, x_val, t_val,
                      epochs=epochs, mini_batch_size=100,
                      optimizer='sgd', optimizer_param={'lr': lr}, verbose=False)
    trainer.train()

    return trainer.test_acc_list, trainer.train_acc_list

# hyperparameter random search
optimization_trial = 10
results_val = {}
results_train = {}

for _ in range(optimization_trial):
    lr = 10 ** np.random.uniform(-6, -2) # 균일분포.
    weight_decay = 10 ** np.random.uniform(-8, -4)

    val_acc_list, train_acc_list =__train(lr, weight_decay)

    print("val_acc :"+str(val_acc_list[-1]))
    key = "lr:"+str(lr)+", weight_decay:"+str(weight_decay)
    results_val[key] = val_acc_list
    results_train[key] = train_acc_list


# 그래프 그리기========================================================
print("=========== Hyper-Parameter Optimization Result ===========")
graph_draw_num = 20
col_num = 5
row_num = int(np.ceil(graph_draw_num / col_num)) # np.ceil :반올림
i = 0

# sorted : 소팅, key : 일정 기준으로(여기서는 val_acc_list의 제일 마지막 원소, 즉 제일 마지막 acc)
# reverse : 내림차순
for key, val_acc_list in sorted(results_val.items(), key=lambda x:x[1][-1], reverse=True):
    print("Best-" + str(i+1) + "(val acc:" + str(val_acc_list[-1]) + ") | " + key)

    plt.subplot(row_num, col_num, i+1)
    plt.title("Best-" + str(i+1))
    plt.ylim(0.0, 1.0)
    if i % 5: plt.yticks([])
    plt.xticks([])
    x = np.arange(len(val_acc_list))
    plt.plot(x, val_acc_list)
    plt.plot(x, results_train[key], "--")
    i += 1

    if i >= graph_draw_num:
        break


plt.show()