import torch

data =torch.randn(2,3)
print(data.dtype)
print(data.type(torch.IntTensor).dtype)
print(data.int().dtype)