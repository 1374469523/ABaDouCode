# ============================================================
# 演示主题：激活函数（Sigmoid / Tanh / ReLU）与 Softmax
# 核心结论：
#   1. 激活函数给神经网络引入“非线性”。没有它，无论堆多少层
#      全连接层，本质上都等价于一个线性变换，无法解决
#      XOR（异或）这类线性不可分问题。
#   2. Sigmoid：值域 (0,1)，适合作为概率输出；但容易梯度饱和（两端导数趋近 0），
#      深层网络中会导致“梯度消失”。
#   3. Tanh：值域 (-1,1)，输出以 0 为中心，效果通常优于 Sigmoid，但同样会梯度饱和。
#   4. ReLU：x>0 时梯度恒为 1，x<0 时输出 0，计算快、能缓解梯度消失，
#      是目前默认首选；缺点是负数区间梯度为 0（神经元“死亡”）。
#   5. Softmax：把一组实数“分数”归一化成概率，总和为 1，常用于分类输出层。
# ============================================================

import torch
import matplotlib.pyplot as plt

# # 函数
# x = torch.linspace(-20,20,1000)          # 在 [-20, 20] 上均匀取 1000 个点，作为自变量
# # y = torch.sigmoid(x)                     # 画出 Sigmoid 曲线：S 形，值域 (0,1)
# # y = torch.tanh(x)                        # 画出 Tanh 曲线：S 形，值域 (-1,1)，关于原点对称
# y = torch.relu(x)                        # 画出 ReLU 曲线：x>0 时 y=x，x<=0 时 y=0（折线）
# plt.plot(x,y)
# plt.grid()                               # 显示网格，方便观察曲线形状
# plt.show()
#
# # 导函数
# x = torch.linspace(-20,20,1000,requires_grad=True)  # requires_grad=True：告诉 PyTorch 要记录梯度
# # torch.sigmoid(x).sum().backward()        # 对 Sigmoid 整体求和后反向传播，得到逐点导数
# # torch.tanh(x).sum().backward()           # 对 Tanh 反向传播，得到逐点导数
# torch.relu(x).sum().backward()            # 对 ReLU 反向传播，得到逐点导数
# plt.plot(x.detach(),x.grad)               # x.detach() 去掉梯度信息用于画图；x.grad 保存着导数
# plt.grid()
# plt.show()
# # 观察要点：Sigmoid/Tanh 在中间导数接近 1、两端趋近 0（饱和，梯度消失）；
# #          ReLU 在正半轴导数恒为 1，负半轴为 0。

# ---- Softmax 演示：把一组原始分数变成概率分布 ----
scores = torch.tensor([0.2, 0.02, 0.15, 0.15, 1.3, 0.5, 0.06, 1.1, 0.05, 3.75])
# 这 10 个数字可理解为 10 个类别的“未归一化分数”（logits），数值越大表示越像该类。
# dim=0 表示沿着第 0 维（10 个分数这一维度）做归一化。
print(torch.softmax(scores, dim=0))
# 输出结果：每个值都在 0~1 之间、全部相加等于 1；分数最大的 3.75 对应的概率最大，
# 说明 Softmax 会“放大”大的分数、压制小的分数，适合做多分类的最终输出。
