import torch
import torch.nn as nn

# torch.random.manual_seed(22)
input = torch.randn([1,4])
layer = nn.Linear(in_features=4, out_features=5)
y = layer(input)
print(y)

dropout =nn.Dropout(p=0.75)
out =dropout(y)
print(out)
