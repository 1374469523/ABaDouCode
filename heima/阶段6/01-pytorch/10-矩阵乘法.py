import torch

torch.random.manual_seed(22)
data1 = torch.randint(0,10,[3,4])
print(data1)

torch.random.manual_seed(23)
data2 = torch.randint(0,10,[4,5])
print(data2)

print(data1 @ data2)
print(torch.matmul(data1, data2))
