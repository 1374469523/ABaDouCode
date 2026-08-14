# ============================================================
# 演示主题：池化层（MaxPool2d / AvgPool2d）
#
# 核心结论：
#   1. 池化也是用一个小窗口在特征图上滑动，但池化层没有可学习的参数，
#      它只是对窗口内的值做"取最大值"(MaxPool) 或"取平均值"(AvgPool)；
#   2. 池化的作用：降低特征图的尺寸（降维）→ 减少参数量、防止过拟合；
#      同时让模型对微小的位置变化更不敏感（平移不变性）；
#   3. 输出尺寸公式（padding=0 时）：输出H = (输入H - kernel) / stride + 1。
# ============================================================
import torch          # 导入 PyTorch 核心库
import torch.nn as nn # 导入神经网络模块，包含 MaxPool2d、AvgPool2d 等层

# 下面是被注释掉的 1 通道示例：一个 3x3 的特征图
# inputs = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]]]).float()
# print(inputs.shape)
# 这里构造一个形状为 (1, 3, 3, 3) 的输入：
#   第1维 1     ：批量大小 batch，这里只有 1 张图
#   第2维 3     ：通道数，相当于有 3 个特征图
#   后两维 3x3  ：每个特征图的高和宽
inputs = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                       [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
                       [[11, 22, 33], [44, 55, 66], [77, 88,99]]]).float()
print(inputs.shape)  # 打印 (1, 3, 3, 3)，确认输入维度

# 最大池化：窗口 kernel_size=2x2，stride=1，在每个窗口内取最大值
pooling = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
print(pooling(inputs))  # 输出形状 (1, 3, 2, 2)，尺寸由 3x3 缩小为 2x2
pooling = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)  # 平均池化：每个窗口内取平均值
print(pooling(inputs))  # 与最大池化输出尺寸相同，但数值是窗口内的平均值
