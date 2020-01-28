import numpy as np

# loss func
def mse(x, y):
    return 0.5 * np.sum((x-y)**2)

def cross_entropy(x,y):
    delta = 1e-7
    return -np.sum(y*np.log(x+delta))


def cross_entropy_batch(x,y):
    delta = 1e-7
    if np.ndim(x)==1:
        x = x.reshape(1,-1)
        y = y.reshape(1,-1)

    batch_size = x.shape[0]

    return -np.sum(y*np.log(x+delta)) / batch_size


y = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

x = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
result = cross_entropy(np.array(x),np.array(y))
print(result)

x = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
result = cross_entropy(np.array(x), np.array(y))
print(result)