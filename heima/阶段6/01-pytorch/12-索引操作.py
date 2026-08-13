import torch
#
torch.random.manual_seed(22)
data =torch.randint(0,10,[4,5])
print(data)

# print(data[2,3])

# print(data[[1, 2], [2, 4]])
#
# print(data[[[1], [3]], [1, 3]])

# print(data[:2, 1:4:2])

index = data[2]>2
print(index)
print(data[:,index])


# torch.random.manual_seed(22)
# data =torch.randint(0,10,[3,4,5])
# print(data[1,2,2])
