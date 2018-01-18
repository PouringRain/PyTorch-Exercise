# encoding = utf8

import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn

xy = np.loadtxt('/home/shengjian/下载/diabetes.csv.gz', delimiter=',', dtype=np.float32)
x = Variable(torch.from_numpy(xy[:,0:-1]))
y = Variable(torch.from_numpy(xy[:, -1:]))


print(x.data.shape)
print(y.data.shape)

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(8, 6)
        self.l2 = nn.Linear(6, 4)
        self.l3 = nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        out = self.sigmoid(self.l3(out2))

        return out

model = Model()
print(model)

loss_function = nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    y_pred = model(x)
    loss = loss_function(y_pred, y)
    print(epoch, loss.data[0])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

