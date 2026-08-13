import torch

torch.random.manual_seed(22)
data1 = torch.randint(0,10,[3,3])
print(data1)

torch.random.manual_seed(23)
data2 = torch.randint(0,10,[2,3])
print(data2)

print(torch.mul(data1, data2))

print(data1 * data2)