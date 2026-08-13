# one-hot编码
# Word2vec
# Word Embedding
import torch

# 几个冒号就是几维 大写是类 小写是对象
a = torch.randn(2, 3, 4)
print(a)
print(a[:1].shape)  # dim = 3([1, 3, 4])
print(a[:1, :2])
print(a[:1, :2].shape)  # dim = 3 ([1,2,4])
print(a[1, :2, 3])
print(a[1, :2, 3].shape)  # dim = 1 ([2])

print(a[:, 2, :2])
print(a[:, 2, :2].shape)  # dim = 2 ([2,2]) 错误应该直接报错
