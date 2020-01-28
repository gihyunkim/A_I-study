import sys, os
sys.path.append(os.pardir)
from minst_2_02 import get_data, init_network, predict
import numpy as np

x, y = get_data()
network = init_network()
batch = 100
accuracy_cnt = 0
for i in range(0, len(x),batch):
    batch_x = x[i:i+batch]
    pred = predict(network, batch_x)
    p = np.argmax(pred)

    batch_y = y[i:i+batch]

    accuracy_cnt += np.sum( p==batch_y)

print(accuracy_cnt / len(x) * 100)