# --*-- coding:utf-8 --*--
"""
作者：徐柯炎
日期：2023年10月13日
"""
import numpy as np

a = np.array([[1, 2], [2, 3], [2, 5]])
c = np.tile(a, (2, 1))
# print(c)
b = a[1]
b = [3, 4]
a[1] = b
# print(a)
max_index = np.argmax(a, axis=1)
# print(b)
# print(max_index)
# print(np.diag(a[1]))
a = np.array([1, 2])
b = np.array([[1], [2]])
c = a + b
print(c)
