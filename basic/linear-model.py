# encoding = utf8

import numpy as np
from matplotlib import pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

def farword(x):
    return x*w

def loss(x,y):
    y_pred = farword(x)
    return (y_pred-y)**2

w_list = []
mse_list = []

for w in np.arange(0,4,.1):
    print('w=', w)
    l_sum = 0
    for x, y in zip(x_data, y_data):
        y_pred = farword(x)
        l = loss(x, y)
        l_sum += l

    print("MSE=", l_sum/3)
    w_list.append(w)
    mse_list.append(l_sum/3)

plt.plot(w_list, mse_list)
plt.xlabel('w')
plt.ylabel('mse_loss')
plt.show()


