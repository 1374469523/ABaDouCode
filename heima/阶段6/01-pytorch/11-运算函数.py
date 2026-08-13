import torch

torch.random.manual_seed(22)
data = torch.randn([3,4])
print(data)

print(data.mean())
print(data.mean(dim=0))
print(data.mean(dim=1))

print(data.sum())
print(data.sum(dim=0))
print(data.sum(dim=1))

print(data.sqrt())
print(torch.pow(data, 2))
print(torch.pow(2, data))
print(data.exp())

print(data.log())
print(data.log2())
print(data.log10())