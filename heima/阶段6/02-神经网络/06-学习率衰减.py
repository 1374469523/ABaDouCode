# ============================================================
# 演示主题：学习率衰减（Learning Rate Scheduler）
# 核心结论：
#   1. 学习率 lr 决定每步参数更新的“步子”大小。
#        - 学习率过大：震荡剧烈，甚至在最小值附近来回跳动、无法收敛（发散）。
#        - 学习率过小：收敛极慢，训练半天还在原地踏步。
#   2. 学习率衰减的思想：训练初期用较大的 lr 快速逼近最优区域，
#      训练后期用较小的 lr 精调，从而既快又稳地收敛。
#   3. 常用策略：
#        - StepLR：每隔 step_size 轮，lr 乘以 gamma（阶梯式下降）。
#        - MultiStepLR：在指定的里程碑轮次（milestones）处，lr 乘以 gamma。
#        - ExponentialLR：每一轮 lr 都乘以 gamma（指数式连续衰减）。
#   4. 使用方式：每训练完一个 epoch 调用一次 scheduler.step() 更新 lr；
#      optimizer.step() 和 scheduler.step() 是两回事，别混淆。
# ============================================================

import torch
import matplotlib.pyplot as plt

# 参数初始化
lr0 = 0.1        # 初始学习率：衰减前的起点
iter = 100       # 每个 epoch 内部迭代（更新参数）的次数
epoches = 200    # 训练总轮数（epoch）

# 网络数据初始化
x = torch.tensor([1.0])                            # 输入
w = torch.tensor([1.0],requires_grad=True)         # 待优化的参数 w
y = torch.tensor([1.0])                            # 真实值

# 优化器
optimizer=torch.optim.SGD([w],lr=lr0,momentum=0.9)
# SGD 优化器：momentum=0.9 加入动量，让更新更平稳；初始学习率取 lr0。

# 学习率策略（三选一，可自行切换观察曲线差异）
# scheduler_lr=torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.8)
# 上面：StepLR，每 20 轮，学习率乘以 0.8（阶梯下降）。
# scheduler_lr=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[20,60,90,135,180],gamma=0.8)
# 上面：MultiStepLR，在 [20,60,90,135,180] 这些轮各乘一次 0.8（手动指定下降点）。
scheduler_lr=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.99)
# 上面：ExponentialLR，每一轮学习率都乘以 0.99（指数式连续衰减）。

# 遍历轮次：记录每一轮的学习率，画出来观察衰减曲线
epoch_list = []   # 存放轮次序号（横轴）
lr_list =[]       # 存放每一轮对应的学习率（纵轴）
for epcoh in range(epoches):
    lr_list.append(scheduler_lr.get_last_lr())   # 记录当前学习率（get_last_lr 返回该轮的 lr 列表）
    epoch_list.append(epcoh)                     # 记录轮次
    # 遍历batch：这里用固定循环 iter 次代替真实数据批次
    for i in range(iter):
        # 计算损失：loss = 0.5 * (w*x - y)^2，目标是最小化它
        loss = ((w*x-y)**2)*0.5
        # 更新参数（标准的四步：清零梯度 -> 反向传播 -> 更新）
        optimizer.zero_grad()   # 清零上一次梯度
        loss.backward()         # 反向传播计算梯度
        optimizer.step()        # 用当前学习率更新参数 w
    # 更新lr：每个 epoch 结束后调用 scheduler 让学习率按策略衰减一次
    scheduler_lr.step()

# 绘制结果：学习率随轮次的变化曲线
plt.plot(epoch_list,lr_list)   # 横轴轮次、纵轴学习率
plt.grid()                     # 显示网格
plt.show()
# 观察：指数衰减曲线最平滑；StepLR 是阶梯状；MultiStepLR 只在指定轮次骤降。
