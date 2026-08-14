# ============================================================
# 演示主题：RNN 的基本用法与张量形状
#   用一个最简单的例子，看 RNN 如何一次性接收"整个序列"并输出结果。
#
# 核心结论：
#   1. RNN 是"按时间步"工作的：把序列一个接一个地喂进去，
#      每步输入结合上一步的"隐藏状态"，得到本步输出和新的隐藏状态。
#   2. 默认 batch_first=False 时，输入 x 的形状是
#      (seq_len, batch, input_size)，即 (时间步数, 批次大小, 每个输入维度)。
#   3. 输出 y 的形状是 (seq_len, batch, hidden_size)，是"每一个时间步"的输出；
#      hn 的形状是 (num_layers, batch, hidden_size)，是"最后一层最后一个时间步"
#      的隐藏状态，常被当作整句话的"总结/语义向量"。
#   4. 初始隐藏状态 h0 用全 0 表示：从"空白记忆"开始看序列。
# ============================================================
import torch
import torch.nn as nn


# 创建 RNN 层：
#   input_size=128   -> 每个时间步输入向量的维度
#   hidden_size=64   -> 隐藏状态（记忆）的维度
#   num_layers=10    -> 堆叠 10 层 RNN（层数越多模型能力越强，但也更难训练）
#   注意：这里没有写 batch_first=True，所以默认是 batch_first=False，
#         输入形状要按 (seq_len, batch, input_size) 来提供
rnn = nn.RNN(input_size=128,hidden_size=64,num_layers=10)
# 随机生成输入序列 x：
#   batch_first=False 时形状 = (seq_len, batch, input_size)
#   = (时间步 12, 批次 24, 每步特征 128)，即"24 条长度为 12 的序列"
x = torch.randn(12,24,128)
# 初始隐藏状态 h0：形状 = (num_layers, batch, hidden_size) = (10, 24, 64)
#   全 0 表示初始时"没有任何记忆"
h = torch.zeros(10,24,64)
# 把整个序列一次性交给 RNN：
#   y  = 所有时间步的输出，形状 (seq_len, batch, hidden_size) = (12, 24, 64)
#   hn = 最后一层、最后一步的隐藏状态，形状仍是 (num_layers, batch, hidden_size) = (10, 24, 64)
y,hn=rnn(x,h)
print(y.shape)   # 期望输出 torch.Size([12, 24, 64])
print(h.shape)   # 期望输出 torch.Size([10, 24, 64])