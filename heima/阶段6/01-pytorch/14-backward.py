# ============================================================
# 文件主题：自动求导（autograd）—— backward() 与梯度 grad
# 核心结论：
#   1. requires_grad=True 表示「这个参数需要计算梯度」，PyTorch 会自动记录
#      一切由它参与的计算，形成计算图
#   2. loss.backward()：从 loss 反向传播（链式法则），自动算出 loss 对每个
#      requires_grad=True 参数的偏导数（梯度），存到各自的 .grad 属性
#   3. 梯度存放在 w.grad / b.grad 中，形状与 w / b 完全一致，供后续优化器更新参数
# 通俗解释「梯度」：梯度 = 某个参数对最终损失的影响大小和方向。
#   梯度越大，说明调整该参数对减小 loss 越有效，这正是梯度下降更新参数的依据
# 易混点：只有 requires_grad=True 的张量才会有 .grad；
#         多次 backward 前要先 .zero_grad() 清零（PyTorch 默认累加而不是覆盖）
# ============================================================
import torch
# 数据 特征+目标
# x = torch.tensor(5)    # 被注释：单个标量样本的写法（0 维）
# y = torch.tensor(0.)   # 被注释：单个标量目标的写法
# torch.ones(2,5)：构造输入 x：2 个样本、每个样本 5 个特征，所有元素为 1
x = torch.ones(2,5)
# torch.zeros(2,3)：构造目标 y：2 个样本、每个样本 3 个标签，所有元素为 0
# （是上面那套单样本写法的批量版：样本数 2、输出数 3）
y = torch.zeros(2,3)
# 参数 权重+偏置
# w = torch.tensor(1,requires_grad=True,dtype=torch.float32)   # 被注释：单个权重的写法
# b = torch.tensor(3.,requires_grad=True,dtype=torch.float32)  # 被注释：单个偏置的写法
# torch.randn(5,3,requires_grad=True)：权重矩阵 w，形状 (5,3)（把 5 个特征映射到 3 个输出）
# requires_grad=True 是关键：告诉 PyTorch「这个参数要求梯度」，它是可学习参数
w = torch.randn(5,3,requires_grad=True)
# torch.randn(3,requires_grad=True)：偏置向量 b，形状 (3,)（对应 3 个输出各一个偏置）
b = torch.randn(3,requires_grad=True)
# 预测
# z = x*w + b   # 被注释：单标量情况下才直接 x*w+b；批量场景必须用矩阵乘
# z = torch.matmul(x,w)+b：线性预测。x(2,5) @ w(5,3) → (2,3)，再加偏置 b(3,)（广播到 2 行）
# 数学形式 z = X·W + b，就是一个不带激活函数的线性层
z = torch.matmul(x,w)+b
# 损失
# torch.nn.MSELoss()：创建「均方误差」损失函数对象（mean squared error）
loss =torch.nn.MSELoss()
# loss(z,y)：计算预测 z 与真实 y 之间的均方误差，返回一个标量张量（数值越小预测越准）
loss =loss(z,y)
# 微分
# loss.backward()：反向传播，自动计算 loss 对 w、b 的偏导数（梯度），
# 结果分别写入 w.grad 和 b.grad；这是 PyTorch 用「链式法则」自动完成的，无需手推
loss.backward()
# 梯度
# w.grad：打印 w 的梯度，形状与 w 相同（5,3）。每个值表示 loss 对 w 对应元素的偏导
print(w.grad)
# b.grad：打印 b 的梯度，形状与 b 相同（3,）
print(b.grad)
