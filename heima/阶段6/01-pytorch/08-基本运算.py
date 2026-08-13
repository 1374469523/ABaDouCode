import torch

torch.random.manual_seed(22)
data = torch.randint(0,10,[2,3])
print(data)

print(data.add(10))
print(data.sub(10))
print(data.mul(10))
print(data.div(10))
print(data.neg())
# print(data)