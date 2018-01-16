#coding='utf-8'

import numpy as np

arr = np.arange(12).reshape((3, 4))
print(arr, '\n')

# get first line
print(arr[0, :])

# get from 1 to 2 two lines
print(arr[1:2 :])

# get first col return: as line
print(arr[:, 0])

# get first col return: as col
print(arr[:, :1])

# get data block
print(arr[1:2, 1:3])
print(arr[:,::2])