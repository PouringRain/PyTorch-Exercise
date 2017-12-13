#coding='utf-8'
import numpy as np

a = np.array([
            [11,3,4,34],
            [6,9,14,15],
            [9,55,14,17]
              ])
print(a)

# argmax argmin 矩阵中最小元素和最大元素的索引
print(np.argmax(a))
print(np.argmin(a))

# mean average median中位数
print(np.mean(a))
print(np.average(a))

# cumsum累乘    diff累差
print(np.cumsum(a))
print(np.diff(a))

# nonezero
print(np.nonzero(a))

# sort
# print(np.sort(a, axis=0))

# 转置
print(a)
print(a.T)
print(a.transpose())

# clip 限界
print(np.clip(a,10,20))