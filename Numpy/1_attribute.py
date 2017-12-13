# coding='utf-8'
import numpy as np

'''
ndim：维度
shape：行数和列数
size：元素个数

'''


array_1 = np.array([[1,2,3],
                [4,5,6],
                [7,8,9],
                [10,11,12]])

array_2 = np.array([1,2,3])

array_3 = np.array([[[1,2,3]]])

print(array_1)

print(array_1.size)
print(array_1.shape)
print(array_1.data)
print(array_1.ndim)

print(array_2.ndim)
print(array_3.ndim)