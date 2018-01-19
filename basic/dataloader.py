# encoding = utf8

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn

class DiabetesDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt('/home/shengjian/下载/diabetes.csv.gz', delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x = torch.from_numpy(xy[:,0:-1])
        self.y = torch.from_numpy(xy[:, -1:])

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)

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

for epoch in range(2):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        y_pred = model(inputs)

        loss = loss_function(y_pred, labels)
        print(epoch,i, loss.data[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

