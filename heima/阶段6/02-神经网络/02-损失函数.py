# ============================================================
# 演示主题：损失函数（Loss Function）
# 核心结论：
#   1. 损失函数衡量“模型预测值”与“真实值”的差距，训练的目标就是最小化它。
#   2. 分类任务常用：
#        - CrossEntropyLoss（交叉熵）：多分类标配，内部会自动做 Softmax，
#          target 一般传类别索引（整数，如 0/1/2）。
#        - BCELoss（二元交叉熵）：二分类用，需要配合 Sigmoid 输出，target 为 0/1。
#   3. 回归任务常用：
#        - L1Loss（平均绝对误差 MAE）：对离群点不敏感，但梯度恒为 ±1，不够平滑。
#        - MSELoss（均方误差 MSE）：梯度随误差增大而增大，收敛快；但对离群点敏感。
#        - SmoothL1Loss（平滑 L1，即 Huber Loss）：误差小时像 L2、误差大时像 L1，
#          兼顾两者优点，训练更稳定。
# ============================================================

import torch.nn as nn
import torch

# ---- 多分类：交叉熵损失（演示被注释，可自行取消注释观察） ----
# y_true = torch.tensor([0,1,2],dtype=torch.int64)
# 上面：真实类别索引（0、1、2），CrossEntropyLoss 要求 target 用 int64（长整型）且为索引格式。
# y_true = torch.tensor([[1,0,0],[0,1,0],[0,0,1]],dtype=torch.float32)
# 上面：真实类别用 one-hot 编码（1 表示属于该类别）——这种格式在手动写交叉熵时才需要。
# y_predict = torch.tensor([[18,9,10],[2,14,6],[3,8,16]],dtype=torch.float32)
# 上面：模型输出的原始分数（logits），CrossEntropyLoss 内部会自动做 Softmax，无需手动加。
# loss =nn.CrossEntropyLoss()
# print(loss(y_predict, y_true))
# 说明：分数越大的类别若正好是真实类别，损失越小；这里 18/14/16 正好对应 0/1/2，损失应较小。

# ---- 二分类：二元交叉熵损失（演示被注释，可自行取消注释观察） ----
# y_true = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
# 上面：二分类真实标签（0 或 1），BCELoss 要求 target 为 float32 且值在 [0,1]。
# y_predict = torch.tensor([0.1, 0.9, 0.2, 0.8], dtype=torch.float32)
# 上面：模型输出的概率（需先经过 Sigmoid 压到 0~1 之间），预测值与真实值越接近损失越小。
# loss=nn.BCELoss()
# print(loss(y_predict, y_true))

# ---- 回归：平滑 L1 损失（当前实际运行的代码） ----
y_true = torch.tensor([2.0, 3.0, 1.0], dtype=torch.float32)   # 真实值（连续数值）
y_predict = torch.tensor([1.0, 5.0, 4.0], dtype=torch.float32) # 模型预测值（连续数值）
# loss =nn.L1Loss()          # 若取消注释：计算平均绝对误差 = (|1-2|+|5-3|+|4-1|)/3 = 2.0
# loss =nn.MSELoss()         # 若取消注释：计算均方误差 = ((1^2+2^2+3^2)/3) = 14/3 ≈ 4.67
loss =nn.SmoothL1Loss()      # 平滑 L1：误差绝对值小于 1 时用 0.5*误差^2，否则用 误差-0.5
print(loss(y_predict, y_true))
# 输出即当前损失值。预测与真实相差越大，损失越大，说明模型越差。
