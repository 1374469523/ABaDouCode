import torch.nn as nn
import torch

# y_true = torch.tensor([0,1,2],dtype=torch.int64)
# y_true = torch.tensor([[1,0,0],[0,1,0],[0,0,1]],dtype=torch.float32)
# y_predict = torch.tensor([[18,9,10],[2,14,6],[3,8,16]],dtype=torch.float32)
# loss =nn.CrossEntropyLoss()
# print(loss(y_predict, y_true))

# y_true = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
# y_predict = torch.tensor([0.1, 0.9, 0.2, 0.8], dtype=torch.float32)
# loss=nn.BCELoss()
# print(loss(y_predict, y_true))

y_true = torch.tensor([2.0, 3.0, 1.0], dtype=torch.float32)
y_predict = torch.tensor([1.0, 5.0, 4.0], dtype=torch.float32)
# loss =nn.L1Loss()
# loss =nn.MSELoss()
loss =nn.SmoothL1Loss()
print(loss(y_predict, y_true))
