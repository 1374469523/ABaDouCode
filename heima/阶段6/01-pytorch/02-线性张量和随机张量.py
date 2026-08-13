import torch

print(torch.arange(0, 10, 1))
print(torch.linspace(0, 10, 11))

print(torch.randn(2, 3))
print(torch.random.initial_seed())


torch.random.manual_seed(100)
print(torch.randn(2, 3))
print(torch.random.initial_seed())