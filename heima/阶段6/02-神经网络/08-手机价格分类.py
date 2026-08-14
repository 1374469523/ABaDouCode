# ============================================================
# 演示主题：完整的手机价格档位分类（训练 + 预测 全流程）
# 核心结论：
#   1. 这是一个端到端的深度学习项目模板，包含四步：
#        数据读取/划分 -> 模型构建 -> 模型训练 -> 模型预测与评估
#   2. 本任务是根据 20 个手机硬件特征（如电池、内存、摄像头等），
#      把手机价格分为 0/1/2/3 四个档位（多分类问题）。
#   3. 训练循环的标准四步（每个 batch 都要执行，顺序关键）：
#        y_predict = model(x)    前向传播：算预测值
#        loss = criterion(预,真)  算损失：衡量预测与真实的差距
#        optimizer.zero_grad()   清空上一次梯度（防累加）
#        loss.backward()         反向传播：自动计算每个参数的梯度
#        optimizer.step()        用梯度更新参数
#   4. 预测时：模型输出 4 个分数，用 torch.argmax 取最大的下标作为预测类别；
#      统计预测正确的个数除以总数得到准确率 accuracy。
#   5. 训练完用 state_dict() 保存模型权重，预测时用 load_state_dict() 加载。
# ============================================================

import pandas as pd                                     # 用于读取 CSV 表格数据
from sklearn.model_selection import train_test_split    # 用于划分训练集/测试集
import torch
from torch.utils.data import TensorDataset, DataLoader  # 封装数据集与分批加载
import torch.nn as nn
from torchsummary import summary                        # 打印网络结构

# ============ 1.获取数据 ============
# 1.1 读取数据
data = pd.read_csv('/Users/mac/Desktop/AI20深度学习/02-code/02-神经网络/data/手机价格预测.csv')
# 读取 CSV 文件为 DataFrame；注意这里的路径是 Mac 环境的绝对路径，
# 在自己电脑上运行需要改成数据实际所在的路径。
x = data.iloc[:, :-1]      # 特征：取所有行、所有列除了最后一列（20 个硬件特征）
y = data.iloc[:, -1]       # 标签：取最后一列（价格档位 0/1/2/3）

# 1.2 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# train_test_split：随机打乱并按比例切分，test_size=0.2 表示 20% 做测试集、80% 做训练集。
# print(x_train)             # 可取消注释查看训练特征的前几行
# 把 pandas 数据转成 PyTorch 张量：
x_train = torch.tensor(x_train.values, dtype=torch.float32)   # 特征用 float32（网络计算的默认精度）
x_test = torch.tensor(x_test.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.int64)     # 分类标签用 int64（CrossEntropyLoss 要求）
y_test = torch.tensor(y_test.values, dtype=torch.int64)

# 1.3 封装tensor
train_dataset = TensorDataset(x_train, y_train)   # 把 (特征, 标签) 打包成一一对应的数据集
test_dataset = TensorDataset(x_test, y_test)

# 1.4 构建数据迭代器
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# DataLoader：自动把数据集按 batch_size=8 分成一小批一小批地取。
# shuffle=True 训练时打乱顺序，避免模型学到样本顺序的假规律。
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
# 测试集 shuffle=False：不打乱，保证评估结果稳定可复现。


# ============ 2.模型构建 ============
# 类：定义网络结构
class model(nn.Module):
    # init: 定义网络中的层
    def __init__(self):
        super(model, self).__init__()   # 调用父类初始化，必须写
        self.layer1 = nn.Linear(in_features=20, out_features=64)   # 输入20个特征 -> 64个神经元
        self.layer2 = nn.Linear(in_features=64, out_features=128)  # 64 -> 128，层数越多表达力越强
        self.layer3 = nn.Linear(in_features=128, out_features=4)   # 128 -> 4，输出4个价格档位的分数
        self.dropout = nn.Dropout(p=0.9)  # Dropout：p=0.9 表示训练时随机丢弃 90% 神经元
        # 丢弃比例很大，是为了强烈防止过拟合；实际使用一般取 0.2~0.5。

    # forward: 定义数据流过网络的顺序（前向传播）
    def forward(self, x):
        x = self.layer1(x)        # 第1层线性变换
        x = self.dropout(x)       # Dropout 随机丢弃（只在训练时生效）
        x = torch.relu(x)         # ReLU 激活：引入非线性，负值置 0
        x = self.layer2(x)        # 第2层线性变换
        x = self.dropout(x)       # 再次 Dropout
        x = torch.relu(x)         # 再次 ReLU 激活
        out = self.layer3(x)      # 输出层：得到 4 个类别的原始分数（logits）
        return out                # 返回分数；CrossEntropyLoss 内部会自动做 Softmax，无需手动加


# ============ 3.模型训练 ============
def train():
    phone_model = model()                                  # 创建模型实例
    # 损失
    cri = nn.CrossEntropyLoss()                            # 多分类交叉熵损失，衡量预测与真实档位的差距
    # 优化器
    optimizer =torch.optim.SGD(phone_model.parameters(),lr=0.01)
    # SGD 优化器：phone_model.parameters() 提供模型所有待更新参数，
    # lr=0.01 学习率，决定每步更新幅度（学习率过大易震荡、过小收敛慢）。
    # 遍历
    eopches = 20                                           # 训练总轮数（把所有训练数据过一遍叫 1 个 epoch）
    for epoch in range(eopches):                           # 外层循环：遍历每一个 epoch
        loss_sum=0                                         # 累计本 epoch 所有 batch 的损失，用于打印均值
        sample=0.1                                         # 统计本 epoch 处理的 batch 个数（初始值 0.1 避免除零）
        for x,y in train_dataloader:                       # 内层循环：取一个 batch（8个样本）的 特征x 和 标签y
            y_predict =phone_model(x)                      # 前向传播：输入 x，得到预测分数
            loss =cri(y_predict,y)                         # 计算损失：预测分数 与 真实标签 的差距
            optimizer.zero_grad()                          # ① 清空上一次的梯度（防止跨 batch 累加）
            loss.backward()                                # ② 反向传播：自动算出每个参数的梯度
            optimizer.step()                               # ③ 用梯度更新一次参数（朝减小损失方向走一步）
            loss_sum+=loss.item()                          # 累加当前 batch 的损失数值（.item() 取出标量）
            sample+=1                                      # 已处理的 batch 数 +1
        print(loss_sum/sample)                             # 每个 epoch 结束后，打印平均损失
        # 预期效果：随着 epoch 增加，平均损失应逐步下降，说明模型在学得越来越好。
    torch.save(phone_model.state_dict(),'/Users/mac/Desktop/AI20深度学习/02-code/02-神经网络/data/myphone.pth')
    # 保存训练好的权重：state_dict() 返回 {参数名: 参数值} 字典；
    # 保存后用 .pth 文件存储，方便下次直接加载。注意路径需按自己电脑修改。


# ============ 4.模型预测 ============
def test():
    my_model=model()                                       # 创建同结构的模型
    my_model.load_state_dict(torch.load('/Users/mac/Desktop/AI20深度学习/02-code/02-神经网络/data/myphone.pth'))
    # 加载训练好的权重：torch.load 读入字典，load_state_dict 填入模型。路径需按自己电脑修改。

    correct=0                                              # 记录预测正确的样本总数
    for x,y in test_dataloader:                            # 遍历测试集每个 batch
        y_predict = my_model(x)                            # 前向传播：得到 4 个档位的分数
        y_index =torch.argmax(y_predict,dim=1)             # argmax 取分数最大的下标作为预测类别（dim=1 沿类别维）
        correct+=(y_index==y).sum()                        # 预测值 == 真实值 则为正确，求和得到本 batch 正确数
    acc =correct.item()/len(test_dataset)                  # 正确总数 / 测试样本总数 = 准确率
    print(acc)                                             # 打印测试集准确率（如 0.6 表示 60% 预测正确）




if __name__ == '__main__':
    my_model = model()                                     # 创建模型用于查看结构
    summary(my_model, input_size=(20,), batch_size=10)     # 打印网络各层输出形状与参数量
    # train()     # 若要训练模型，取消这行注释运行（会覆盖保存的权重）
    test()       # 调用测试函数：加载已有权重并评估准确率
