# encoding = utf8

import torch
from torch.autograd import Variable
from matplotlib import pyplot as plt

LR = 0.01
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = Variable(torch.Tensor([1.0]), requires_grad=True)

def farword(x):
    return x*w

def loss(x,y):
    y_pred = farword(x)
    return (y_pred-y)**2

w_list = []
mse_list = []


print(farword(4))

for epoch in range(100):
    # print('w=', w)
    l_sum = 0
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l_sum += l.data.numpy()
        l.backward()
        print("grad:", w.grad.data[0])
        w.data = w.data - 0.01 * w.grad.data

        w.grad.data.zero_()
    print("l_sum=", l_sum)
    print("MSE=", l_sum/3)
    print("w.data--------------", type(w.data), w.data)
    w_list.append(w.data[0])
    mse_list.append(l_sum/3)

print(farword(4))

plt.plot(w_list, mse_list)
plt.xlabel('w')
plt.ylabel('mse_loss')
plt.show()

