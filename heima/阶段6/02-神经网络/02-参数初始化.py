# ============================================================
# 演示主题：参数初始化（Parameter Initialization）
# 核心结论：
#   1. 权重初始化决定了训练开始时的“起点”。如果初始值全相同（如全 0 或全 1），
#      同一层的所有神经元会得到相同的梯度，导致“对称性问题”：
#      所有神经元学习到完全一样的东西，网络表达能力退化成单神经元。
#   2. 合适的初始化（如 Kaiming / Xavier）能保证信号在前向传播时
#      既不被放大爆炸、也不被缩小消失，让梯度训练更稳定。
#   3. PyTorch 中 nn.Linear 默认已带有合理的初始化；这里演示各种 nn.init.xxx
#      手动初始化写法，供学习和对比。
# ============================================================

import torch
import torch.nn as nn

# torch.random.manual_seed(22)
# 上面的语句用于固定随机种子，保证每次运行结果一致（复现实验用）。

# nn.Linear(in_features, out_features)：
#   创建一个全连接（线性）层，输入特征维度为 3，输出特征维度为 2。
#   层内部自动创建两个参数：
#     - weight（权重矩阵，形状 [out_features, in_features] = [2, 3]）
#     - bias（偏置向量，形状 [out_features] = [2]，默认开启 bias=True）
linear = nn.Linear(in_features=3,out_features=2)

# 下面依次演示 8 种不同的权重初始化方式（一行执行后，weight 会被重新填充）：
nn.init.zeros_(linear.weight)            # 全 0 初始化：所有权重变为 0（会导致对称性，一般不单独用）
nn.init.ones_(linear.weight)             # 全 1 初始化：所有权重变为 1（同样存在对称性问题）
nn.init.constant_(linear.weight,100)     # 常量初始化：所有权重都设为常量 100（过大，易造成梯度爆炸）
nn.init.normal_(linear.weight,mean=0,std=1)   # 正态分布初始化：从 N(mean=0, std=1) 中随机采样
nn.init.uniform_(linear.weight)          # 均匀分布初始化：从默认区间 [-1/sqrt(n), 1/sqrt(n)] 随机采样（n 为输入维度）
nn.init.kaiming_normal_(linear.weight)   # Kaiming 正态初始化：适配 ReLU 激活的方差缩放，缓解梯度消失/爆炸
nn.init.kaiming_uniform_(linear.weight)  # Kaiming 均匀初始化：Kaiming 的均匀分布版本（与 ReLU 搭配推荐）
nn.init.xavier_normal_(linear.weight)    # Xavier 正态初始化：适配 Sigmoid/Tanh，按输入输出维度折中缩放
nn.init.xavier_uniform_(linear.weight)   # Xavier 均匀初始化：Xavier 的均匀分布版本（与 Sigmoid/Tanh 搭配推荐）

print(linear.weight.data)
# 打印当前权重（上面最后一个执行的是 xavier_uniform_，所以显示的是它的结果）。
# 注意：data 会返回脱离梯度记录的普通张量，方便查看数值。
# print(linear.bias.data)
# 如需查看偏置向量，可取消上面的注释；bias 默认也会被初始化（通常为均匀分布的较小数值）。
