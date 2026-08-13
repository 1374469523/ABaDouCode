import torch
import torch.nn as nn

# inputs = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]]]).float()
# print(inputs.shape)
inputs = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                       [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
                       [[11, 22, 33], [44, 55, 66], [77, 88,99]]]).float()
print(inputs.shape)

pooling = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
print(pooling(inputs))
pooling = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
print(pooling(inputs))
