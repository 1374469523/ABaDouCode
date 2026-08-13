import torch

data = torch.randn(2,3)
print(torch.ones(4, 5))
print(torch.ones_like(data))
print(torch.zeros(4, 5))
print(torch.zeros_like(data))
print(torch.full([4, 5], 100))
print(torch.full_like(data, 200))