# ============================================================
# 演示主题：梯度下降优化器（SGD / Adagrad / RMSprop / Adam）
# 核心结论：
#   1. 训练一个参数的一步标准流程（四步，顺序很重要）：
#        loss.backward()      计算梯度（反向传播，把每个参数的导数算出来）
#        optimizer.zero_grad() 清空上一次的梯度（必须先清零，否则梯度会累加）
#        optimizer.step()     用梯度更新参数
#      注意：此处把 zero_grad 放在最前面也能工作，但更规范的顺序是
#      在 backward 之前先清零，避免旧梯度污染本次更新。
#   2. 各优化器特点：
#        - SGD：最朴素的梯度下降，更新慢、容易震荡。
#        - SGD + momentum：加入“动量”，利用历史梯度方向，抑制震荡、加速收敛。
#        - Adagrad：为每个参数自适应学习率，越稀疏的参数更新的步子越大。
#        - RMSprop：用指数加权平均的梯度平方来归一化学习率，缓解 Adagrad 学习率衰减过快的缺点。
#        - Adam：结合动量（一阶矩）与 RMSprop（二阶矩），并做偏差校正，
#          是目前最常用的默认优化器，参数少、效果好、收敛快。
#   3. requires_grad=True 表示需要对该张量计算并保存梯度；
#      w.detach() 返回一个“脱离计算图”的新张量，用来查看更新后的数值。
# ============================================================

import torch

# w 是需要被优化的参数：requires_grad=True 表示 PyTorch 会记录它的计算图并求梯度
w = torch.tensor([1.0],requires_grad=True,dtype=torch.float32)
# 构造损失函数：loss = 0.5 * w^2，该函数在 w=0 处取得最小值
loss = ((w**2)*0.5).sum()

# 下面依次演示 4 种优化器（各自运行一次，更新一次参数后观察结果）：
# optimizer = torch.optim.SGD([w],lr=0.01,momentum=0.9)
# 上面：SGD 带动量，lr=0.01 学习率，momentum=0.9 动量系数（0.9 表示保留 90% 的历史更新方向）。
# optimizer = torch.optim.Adagrad([w],lr=0.01)
# 上面：Adagrad，为每个参数自适应学习率。
# optimizer = torch.optim.RMSprop([w],lr=0.01,alpha=0.9)
# 上面：RMSprop，alpha=0.9 是梯度平方的指数衰减系数。
optimizer = torch.optim.Adam([w],lr=0.01,betas=[0.9,0.99])
# 上面：Adam，lr=0.01 学习率，betas=[0.9,0.99] 分别是一阶矩（动量）和二阶矩（梯度平方）的衰减系数。

# ---- 第一步参数更新 ----
optimizer.zero_grad()   # 清空梯度（防止与上一轮梯度累加）
loss.backward()         # 反向传播：计算 loss 对 w 的梯度（数学上应为 w，此处 w=1.0，所以 grad≈1.0）
optimizer.step()        # 更新参数：w = w - lr * 梯度（不同的优化器算法不同，但都让 w 朝减小 loss 的方向走）
print(w.grad)           # 打印梯度值（更新前算出的梯度，仍保存在 w.grad 中）
print(w.detach())       # w.detach()：脱离计算图查看更新后的数值（比初始值 1.0 更接近最优解 0）

# ---- 第二步参数更新（继续向最优解逼近） ----
loss = ((w**2)*0.5).sum()   # 用更新后的 w 重新计算损失
optimizer.zero_grad()       # 先清零上一轮梯度
loss.backward()             # 再次反向传播，得到新的梯度
optimizer.step()            # 再次更新参数
print(w.grad)               # 打印新的梯度
print(w.detach())           # 打印更新后的 w（多次迭代后应越来越接近 0，即损失最小点）
