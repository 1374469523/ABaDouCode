import torch
# 数据 特征+目标
# x = torch.tensor(5)
# y = torch.tensor(0.)
x = torch.ones(2,5)
y = torch.zeros(2,3)
# 参数 权重+偏置
# w = torch.tensor(1,requires_grad=True,dtype=torch.float32)
# b = torch.tensor(3.,requires_grad=True,dtype=torch.float32)
w = torch.randn(5,3,requires_grad=True)
b = torch.randn(3,requires_grad=True)
# 预测
# z = x*w + b
z = torch.matmul(x,w)+b
# 损失
loss =torch.nn.MSELoss()
loss =loss(z,y)
# 微分
loss.backward()
# 梯度
print(w.grad)
print(b.grad)
