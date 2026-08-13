import torch
import matplotlib.pyplot as plt

# # 函数
# x = torch.linspace(-20,20,1000)
# # y = torch.sigmoid(x)
# # y = torch.tanh(x)
# y = torch.relu(x)
# plt.plot(x,y)
# plt.grid()
# plt.show()
#
# # 导函数
# x = torch.linspace(-20,20,1000,requires_grad=True)
# # torch.sigmoid(x).sum().backward()
# # torch.tanh(x).sum().backward()
# torch.relu(x).sum().backward()
# plt.plot(x.detach(),x.grad)
# plt.grid()
# plt.show()

scores = torch.tensor([0.2, 0.02, 0.15, 0.15, 1.3, 0.5, 0.06, 1.1, 0.05, 3.75])
print(torch.softmax(scores, dim=0))
