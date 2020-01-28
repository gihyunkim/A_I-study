# 수치적 미분
'''
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / 2*h
'''


# y = 0.01x^2 + 0.1x
def function_1(x):
    return 0.01 * x**2 + 0.1*x

import numpy as np

import matplotlib.pyplot as plt

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)

# plt.plot(x,y)
# plt.show()

def function_2(x):
    return x[0]**2 + x[1]**2


# 변수 두개 이상의 미분 구하기.
def numerical_diff(f,x):
    grad = np.zeros_like(x)
    h = 1e-4
    for i in range(x.size):
        tmp = x[i]
        x[i] = tmp+h
        fxh1 = f(x)

        x[i] = tmp-h
        fxh2 = f(x)

        grad[i] = (fxh1 - fxh2) / (2*h)
        x[i] = tmp
    return grad


# gradient_descent : w - lr * df/dw
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_diff(f,x)
        # print("grad: "+str(grad))
        x -= (lr*grad)
        # print("x: "+str(x))
    return x


init_x = np.array([-3.0, 4.0])
grad = gradient_descent(function_2, init_x=init_x,lr=0.1, step_num=100)
print(grad)

# too big learning rate
init_x = np.array([-3.0, 4.0])
grad = gradient_descent(function_2, init_x=init_x,lr=10, step_num=100)
print(grad)

# too small learning rate
init_x = np.array([-3.0, 4.0])
grad = gradient_descent(function_2, init_x=init_x,lr=1e-10, step_num=100)
print(grad)
