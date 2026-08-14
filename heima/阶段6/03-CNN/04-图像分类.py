# ============================================================
# 演示主题：用卷积神经网络（CNN）对 CIFAR10 彩色图片做 10 分类
#
# 核心结论 / 整体流程：
#   1. 数据：CIFAR10 包含 10 类彩色小图（每张 32x32x3），
#      训练集 50000 张、测试集 10000 张；
#   2. 模型：两个"卷积+池化"块提取特征，再接 3 个全连接层（Linear）做分类，
#      最终输出 10 个类别的分数；
#   3. 训练：用 CrossEntropyLoss 计算损失、Adam 优化器更新参数；
#   4. 测试：加载保存好的模型参数，用 argmax 取分数最大的下标作为预测类别，
#      与真实标签比较，统计正确率 acc。
# ============================================================
from torchvision.datasets import CIFAR10      # 导入 CIFAR10 数据集（torchvision 自带）
from torchvision.transforms import Compose, ToTensor  # 导入数据预处理工具
import matplotlib.pyplot as plt               # 绘图库，用于可视化数据集样本
import torch                                  # PyTorch 核心库
import torch.nn as nn                         # 神经网络模块，包含 Conv2d / Linear 等层
from torchsummary import summary              # 用于打印模型的每层结构和参数量
from torch.utils.data import DataLoader       # 数据加载器，用于按批读取数据

# 数据获取
# CIFAR10(root, train, transform) 参数说明：
#   root='data'     ：数据集下载/保存到本地 data 文件夹
#   train=True/False：True 取训练集，False 取测试集
#   transform        ：对每张图片做的预处理
# Compose([ToTensor()])：按顺序执行一组变换，这里只有 ToTensor()
# ToTensor()        ：把图片转成 tensor，并把像素值从 0~255 归一化到 0~1，
#                     归一化后数值统一，能让模型训练更稳定
train_data = CIFAR10(root='data', train=True, transform=Compose([ToTensor()]))
test_data = CIFAR10(root='data', train=False, transform=Compose([ToTensor()]))


# 下面是被注释掉的探索数据集的代码，方便理解数据结构：
# print(test_data.data.shape)      # 测试集图片数组形状 (10000, 32, 32, 3)
# print(train_data.data.shape)     # 训练集图片数组形状 (50000, 32, 32, 3)
# print(train_data.classes)        # 10 个类别名，如 'airplane'、'cat' 等
# print(train_data.class_to_idx)   # 类别名 → 类别编号(0~9) 的映射字典
#
# plt.imshow(train_data.data[100]) # 显示第 100 张训练图片
# print(train_data.targets[100])   # 打印该图片的类别编号（0~9）
# plt.show()
# 模型构建：自定义一个 CNN 分类网络，继承 nn.Module
class imgClassification(nn.Module):
    # 初始化：在这里定义网络需要用到的所有层
    def __init__(self):
        super(imgClassification, self).__init__()
        # 卷积层1：输入是 32x32 的 RGB 彩图（3 通道），用 6 个 3x3 卷积核提取特征
        # out_channels=6 → 输出 6 个特征图；stride=1, padding=0 → 输出尺寸 (32-3+0)/1+1 = 30，即 30x30
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
        # 池化层1：最大池化，2x2 窗口、步长 2 → 尺寸从 30x30 缩小一半到 15x15，
        # 减少参数量、防止过拟合
        self.pooling1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 卷积层2：输入是 6 个特征图（in_channels=6），用 16 个 3x3 卷积核 → 输出 16 个特征图
        # 尺寸：15x15 → (15-3+0)/1+1 = 13，即 13x13
        self.layer2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=0)
        # 池化层2：再池化一次，13x13 → (13-2)/2+1 = 6，即 6x6
        self.pooling2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 全连接层3：把特征展平后接全连接层
        # in_features=576 = 16 个特征图 x 6x6，即 16*6*6 = 576
        self.layer3 = nn.Linear(in_features=576, out_features=120)
        # 全连接层4：120 → 60
        self.layer4 = nn.Linear(in_features=120, out_features=60)
        # 输出层：60 → 10，10 对应 CIFAR10 的 10 个类别
        self.out = nn.Linear(in_features=60, out_features=10)

    # 前向传播：定义数据如何按顺序流过这些层
    def forward(self, x):
        x = torch.nn.LeakyReLU(self.layer1(x))  # 卷积 → LeakyReLU 激活函数，增加非线性表达能力
        x = self.pooling1(x)                    # 最大池化降维（15x15）
        x = torch.relu(self.layer2(x))          # 卷积 → ReLU 激活函数
        x = self.pooling2(x)                    # 最大池化降维（6x6）
        x = torch.reshape(x, [x.size(0), -1])   # 展平：把 (N, 16, 6, 6) 变成 (N, 576)，才能接全连接层
        x = torch.relu(self.layer3(x))          # 全连接层3 + ReLU
        x = torch.relu(self.layer4(x))          # 全连接层4 + ReLU
        out = self.out(x)                       # 输出层，得到 10 个类别的分数（logits）
        return out


model = imgClassification()   # 实例化模型，生成一个可训练的网络


# 用 summary 打印网络各层结构（已注释）：
# summary(model,input_size=(3,32,32),batch_size=1)

# ==================== 模型训练 ====================

def train():
    # 损失函数
    cri = nn.CrossEntropyLoss()   # 交叉熵损失，内部自带 softmax，直接输入"类别分数"和真实标签即可
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=[0.9, 0.99])  # Adam 优化器，lr=0.001 为学习率
    # 遍历每个轮次
    epochs = 10                    # 整个训练集被完整训练 10 遍（10 个 epoch）
    loss_mean = []                 # 记录每个 epoch 的平均损失，用于观察训练效果
    for epoch in range(epochs):
        # DataLoader(batch_size, shuffle) 参数说明：
        #   batch_size=2  ：每批取 2 张图，减小显存占用
        #   shuffle=True   ：每个 epoch 打乱数据顺序，让模型学得更稳、防止记住固定顺序
        #   num_workers    ：用几个进程加载数据（这里用默认 0，即主进程加载）
        dataloader= DataLoader(train_data,batch_size=2,shuffle=True)
        # 每个遍历batch
        loss_sum = 0               # 累加本 epoch 所有 batch 的损失
        sample = 0.1               # 统计 batch 个数，用来计算平均损失
        for x,y in dataloader:
            y_predict =model(x)    # 前向传播：输入一批图片 x，得到预测分数 (batch, 10)
            # loss
            loss=cri(y_predict,y)  # 计算预测结果与真实标签 y 的交叉熵损失
            loss_sum+=loss.item()  # loss.item() 取出损失的数值并累加
            sample+=1              # batch 计数加 1
            # 反向传播
            optimizer.zero_grad()  # 清空上一次留下的梯度，避免梯度累加
            loss.backward()        # 反向传播：根据损失自动计算每个参数的梯度
            optimizer.step()       # 用梯度更新一次参数（模型学习的关键一步）
        loss_mean.append(loss_sum/sample)  # 记录本 epoch 的平均损失
        print(loss_sum/sample)     # 打印本 epoch 的平均损失，数值越小说明模型拟合得越好
    print(loss_mean)               # 打印全部 10 个 epoch 的平均损失列表
    # 保存模型权重
    torch.save(model.state_dict(),'/Users/mac/Desktop/AI20深度学习/02-code/03-CNN/model.pth')  # 把训练好的参数保存到文件，方便下次直接加载

# train()   # 训练开关：需要重新训练时，取消注释这一行即可

# ==================== 模型预测（测试） ====================
def test():
    # 测试集 DataLoader：batch_size=8
    # shuffle=False 很重要：测试时不需要打乱顺序，要按序统计正确率
    dataloader = DataLoader(test_data,batch_size=8,shuffle=False)
    # 加载模型
    # 把之前保存的模型参数读回来，赋给当前模型（不需要重新训练）
    model.load_state_dict(torch.load('/Users/mac/Desktop/AI20深度学习/02-code/03-CNN/model.pth'))

    # 遍历数据进行预测
    correct=0     # 记录预测正确的样本个数
    samples = 0   # 记录总共测试的样本个数
    # 测试阶段的标准做法（本代码未显式调用，实际项目中通常加上）：
    #   model.eval()：切换到评估模式，关闭 Dropout/BatchNorm 等训练行为
    #   torch.no_grad()：不计算梯度，加快速度、节省内存
    for x,y in dataloader:
        y_predict = model(x)      # 前向传播得到预测分数 (batch, 10)
        correct += (torch.argmax(y_predict,dim=-1)==y).sum()  # argmax 取分数最大的下标作为预测类别，与真实标签 y 比较，累加正确个数
        samples += len(y)         # 累加本批的样本数量
    acc = correct/(samples+0.000001)  # 正确率 = 正确数 / 总数，加 0.000001 防止除零
    print(acc)                    # 打印模型在测试集上的准确率

test()   # 执行测试：加载模型并评估准确率