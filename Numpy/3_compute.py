#coding='utf-8'
import numpy as np

a = np.array([[1,2,3], [6,4,2]])
b = np.arange(0, 6).reshape((2,3))
print('a:', a)

print('b:', b)
print('---------------------')
# *-->元素对应相乘
print(a+b)
print(a*b)
print(a-b)
print(b<3)
print(b**2)
print(np.sin(a))

# dot-->矩阵标准乘法  axis=0--->按列操作   axis=1--->按行操作
b = np.arange(6).reshape((3,2))
print(np.dot(a, b))
print(a.dot(b))

print(np.min(a))
print(np.max(a))
print(np.sum(a))

print(np.sum(a, axis=0))
print(np.max(a, axis=1))
