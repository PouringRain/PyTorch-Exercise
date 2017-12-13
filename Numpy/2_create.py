#coding='utf-8'
import numpy as np

# ndim = 1
array = np.array([1,2,3])
print(array)

# 指定类型dtype
a = np.array([[1,2,3],
              [4,5,6]
              ], dtype=np.float)
print(a.dtype)

a = np.array([[1,2,3],
              [4,5,6]
              ], dtype=np.float32)
print(a.dtype)
a = np.array([[1,2,3],
              [4,5,6]
              ], dtype=np.int)
print(a.dtype)

a = np.array([[1,2,3],
              [4,5,6]
              ], dtype=np.int32)
print(a.dtype)

# special data
# all zeros
a = np.zeros((2,3))
print(a)

# all ones
a = np.ones((3,4), dtype=np.float)
print(a)

#all empty
a = np.empty((2,3), dtype=np.float32)
print(a)

# arange & reshape & linspace
a = np.arange(1,12,2).reshape((2,3))
print(a)

a = np.linspace(0, 12, 12, dtype=np.int).reshape((3,4))
print(a)