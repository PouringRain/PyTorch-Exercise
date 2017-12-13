#coding='utf-8'
import numpy as np

# 一维索引
a = np.arange(12).reshape((3,4))
print(a)

print(a[1])

# 二维索引
print(a[1,2])
print(a[1][2]) # 等价

print(a[1,1:3]) # 切片 行
print(a[1:3, 1]) # 列

for row in a:
    print(row)

for column in a.T:
    print(column)

print(a)
# 平铺
print(a.flatten())
# flat:迭代器
for item in a.flat:
    print(item)