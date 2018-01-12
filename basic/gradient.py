# encoding = utf8

from matplotlib import pyplot as plt

LR = 0.01
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

def farword(x):
    return x*w

def loss(x,y):
    y_pred = farword(x)
    return (y_pred-y)**2

# compute gradient
def gradient(x, y):
    return 2*x*(x*w-y)

w_list = []
mse_list = []

w = 1.0
print(farword(4))

for epoch in range(100):
    print('w=', w)
    l_sum = 0
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w = w-LR*grad
        l = loss(x, y)
        l_sum += l

    print("MSE=", l_sum/3)
    w_list.append(w)
    mse_list.append(l_sum/3)

print(farword(4))

plt.plot(w_list, mse_list)
plt.xlabel('w')
plt.ylabel('mse_loss')
plt.show()



