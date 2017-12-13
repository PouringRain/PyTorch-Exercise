#coding='utf-8'
import numpy as np

a = np.arange(12).reshape((3, 4))
print(a)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# 横向分割
print(np.split(a, 2, axis=1))
print(np.hsplit(a, 2))

# 纵向分割
print(np.split(a, 3, axis=0))
print(np.vsplit(a, 3))

# 不等量分割
print(np.array_split(a, 3, axis=1))