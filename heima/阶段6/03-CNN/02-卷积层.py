# ============================================================
# 演示主题：卷积层（nn.Conv2d）是如何工作的
#
# 核心结论：
#   1. 卷积核（kernel）是一个小窗口，在特征图上滑动，用"对应位置相乘再求和"
#      的方式提取局部特征（如边缘、纹理），相当于一个自动学习的特征提取器；
#   2. nn.Conv2d 的输入形状是 (批量N, 通道数C, 高H, 宽W)，所以读取图片后
#      要把原数组从 (H, W, C) 调整成 (C, H, W) 才能送入卷积层；
#   3. 输出尺寸计算公式：
#        输出H = (输入H - 卷积核H + 2*padding) / stride + 1
#        输出W = (输入W - 卷积核W + 2*padding) / stride + 1
#   4. 卷积核个数（out_channels）决定输出多少个"特征图"，每个特征图代表一种特征。
# ============================================================
import torch                 # 导入 PyTorch 核心库
import torch.nn as nn        # 导入神经网络模块，包含 nn.Conv2d 等常用层
import matplotlib.pyplot as plt  # 导入 matplotlib，用于读取图片

# 图像读取
img=plt.imread('/Users/mac/Desktop/AI20深度学习/02-code/03-CNN/data/img.jpg')  # 读取图片 → numpy 数组，形状 (H, W, 3)
print(img.shape)  # 打印图片形状，例如 (H, W, 3)，3 表示 RGB 三通道

# 维度调整
img =torch.tensor(img).permute(2,0,1)  # 先转成 tensor，再用 permute(2,0,1) 把 (H,W,C) 调整成 (C,H,W)，符合卷积层的输入格式
img = img.to(torch.float32).unsqueeze(0)  # 转成 float32 类型；unsqueeze(0) 在最前面加一维 → (1, C, H, W)，1 是批量大小 batch=1
print(img.shape)  # 打印调整后的形状 (1, 3, H, W)

# 卷积操作
layer =nn.Conv2d(in_channels=3,out_channels=10,kernel_size=(3,5),stride=(2,3),padding=1)
# 参数说明：
#   in_channels=3      ：输入通道数。因为图片是 RGB 彩色图，所以是 3 通道
#   out_channels=10    ：卷积核个数=10，即输出 10 个特征图
#   kernel_size=(3,5)  ：卷积核窗口是 3x5，决定一次提取特征的局部范围
#   stride=(2,3)       ：步长，卷积核每次滑动的格数，步长越大输出图越小
#   padding=1          ：在输入四周各补 1 圈 0，用来控制输出尺寸
fm =layer(img)   # 把图片送入卷积层，得到特征图 fm
print(fm.shape)  # 打印特征图形状 (1, 10, 输出H, 输出W)，可用尺寸公式验证