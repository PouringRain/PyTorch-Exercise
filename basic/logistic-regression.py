# encoding = utf8

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0], [4.0]]))
y_data = Variable(torch.Tensor([[0.], [0.], [1.], [1.]]))

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        out = F.sigmoid(y_pred)
        return out

model = Model()
print(model)

loss_func = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(500):

    y_pred = model(x_data)
    loss = loss_func(y_pred, y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # print(loss_func.data[0])

test_x = Variable(torch.Tensor([[4.0]]))
y = model(test_x)

print('when x=4 the predicted y is', y[0][0])
print('y=1', y.data[0][0]>0.5)
print('y=0', y.data[0][0]<=0.5)