import torch
import torch.nn as nn

# torch.random.manual_seed(22)

linear = nn.Linear(in_features=3,out_features=2)
nn.init.zeros_(linear.weight)
nn.init.ones_(linear.weight)
nn.init.constant_(linear.weight,100)
nn.init.normal_(linear.weight,mean=0,std=1)
nn.init.uniform_(linear.weight)
nn.init.kaiming_normal_(linear.weight)
nn.init.kaiming_uniform_(linear.weight)
nn.init.xavier_normal_(linear.weight)
nn.init.xavier_uniform_(linear.weight)
print(linear.weight.data)
# print(linear.bias.data)