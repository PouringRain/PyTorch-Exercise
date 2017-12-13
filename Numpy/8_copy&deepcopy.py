#coding='utf-8'
import numpy as np

a = np.arange(4)
# array([0, 1, 2, 3])

b = a
c = a
d = b

print(b is a)  # True
print(c is a)  # True
print(d is a)  # True

b = a.copy()    # deep copy
print(b)
a[3] = 44
print(a)
print(b)
