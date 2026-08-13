import torch
import numpy as np

# print(torch.tensor(100))
# print(torch.tensor([10.5,2.3]))
# data = np.random.randn(10)
# print(data)
# print(torch.tensor(data))

print(torch.Tensor(2, 3))
print(torch.Tensor([100]))
print(torch.Tensor([10.5,2.3]))
data = np.random.randn(10)
print(data)
print(torch.Tensor(data))

print(torch.IntTensor([10.5,2.3]))
print(torch.FloatTensor([100]))