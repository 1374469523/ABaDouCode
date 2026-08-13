import torch

torch.random.manual_seed(22)
data1 =torch.randint(0,10,[4,5,3])
print(data1)

torch.random.manual_seed(23)
data2 =torch.randint(0,10,[4,5,5])
print(data2)
# print(data.size(2))
# print(data.reshape(-1).shape)

# data1 =data.unsqueeze(dim=1).unsqueeze(dim=-1)
# print(data1.shape)
# print(data1.squeeze().shape)

# 目标是[3,4,5,2]
# data1 = torch.transpose(data, 0, 2)
# data2 = torch.transpose(data1, 1, 2)
# data3 = torch.transpose(data2, 2, 3)
# print(data3.shape)
# print(torch.permute(data, [2, 0, 3, 1]).shape)
# print(data.permute([2, 0, 3, 1]).shape)
# print(data.is_contiguous())
# print(data.view(-1).shape)
# data1 = torch.transpose(data,0,1)
# print(data1.is_contiguous())
# data2=data1.contiguous()
# print(data2.view(-1).shape)
#
# if data.is_contiguous():
#     data.view(-1)
# else:
#     data.contiguous().view(-1)

print(torch.cat([data1, data2], dim=2).shape)