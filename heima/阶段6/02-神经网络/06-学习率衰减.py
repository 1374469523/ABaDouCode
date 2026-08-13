import torch
import matplotlib.pyplot as plt

# 参数初始化
lr0 = 0.1
iter = 100
epoches = 200
# 网络数据初始化
x = torch.tensor([1.0])
w = torch.tensor([1.0],requires_grad=True)
y = torch.tensor([1.0])
# 优化器
optimizer=torch.optim.SGD([w],lr=lr0,momentum=0.9)
# 学习率策略
# scheduler_lr=torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.8)
# scheduler_lr=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[20,60,90,135,180],gamma=0.8)
scheduler_lr=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.99)
# 遍历轮次
epoch_list = []
lr_list =[]
for epcoh in range(epoches):
    lr_list.append(scheduler_lr.get_last_lr())
    epoch_list.append(epcoh)
    # 遍历batch
    for i in range(iter):
        # 计算损失
        loss = ((w*x-y)**2)*0.5
        # 更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # 更新lr
    scheduler_lr.step()

# 绘制结果
plt.plot(epoch_list,lr_list)
plt.grid()
plt.show()