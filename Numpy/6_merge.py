#coding='utf-8'
import numpy as np

# np.newaxis <---> None
A = np.array([1, 1, 1])[:, np.newaxis]  # 列上增加维度
B = np.array([2, 2, 2])[:, np.newaxis]

C = np.vstack((A, B))  # vertical stack
D = np.hstack((A, B))  # horizontal stack

print(C.shape)
print(D)

a = np.arange(6)[np.newaxis,:]  # 在行上增加维度
print(a.shape)
print(a)

a = np.arange(6)[np.newaxis,:,np.newaxis]
print(a.shape)

# concatenate合并
a = np.concatenate((A, B), axis=1)   # axis=1 按行
print(a)