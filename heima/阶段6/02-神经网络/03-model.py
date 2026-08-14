# ============================================================
# 演示主题：用 nn.Module 自定义神经网络模型
# 核心结论：
#   1. 在 PyTorch 中定义模型需要继承 nn.Module，并实现两个方法：
#        - __init__：在这里定义层（参数放在这里，会被自动注册、跟踪梯度）。
#        - forward：在这里定义数据的前向传播顺序（如何把输入变成输出）。
#   2. 调用模型时直接写 my_model(x)，PyTorch 会自动调用 forward，
#      千万不要写成 my_model.forward(x)。
#   3. torchsummary 的 summary() 能打印每一层的输出形状和参数量，方便检查网络结构。
#   4. named_parameters() 可以遍历模型里所有参数（权重/偏置）的名字与数值。
# ============================================================

import torch
import torch.nn as nn
from torchsummary import summary

# 类：自定义模型
class model(nn.Module):
    # init: 在这里定义网络中的各层
    def __init__(self):
        super().__init__()  # 调用父类初始化，nn.Module 的初始化必不可少
        # nn.Linear(in_features, out_features)：全连接层
        self.layer1 = nn.Linear(in_features=3,out_features=3)  # 输入3维 -> 输出3维
        nn.init.kaiming_normal_(self.layer1.bias)              # 手动初始化偏置：Kaiming 正态分布
        self.layer2 = nn.Linear(in_features=3,out_features=2)  # 输入3维 -> 输出2维
        nn.init.xavier_uniform_(self.layer2.weight)            # 手动初始化权重：Xavier 均匀分布
        self.out = nn.Linear(in_features=2,out_features=2)     # 输入2维 -> 输出2维
        nn.init.uniform_(self.out.weight)                      # 手动初始化权重：均匀分布

    # forward: 前向传播，定义数据流过各层的顺序
    def forward(self,x):
        x_layer1 = self.layer1(x)            # 第1层：线性变换
        x_layer1=torch.sigmoid(x_layer1)     # 激活函数 Sigmoid：引入非线性，压缩到 (0,1)
        x_layer2 =self.layer2(x_layer1)      # 第2层：线性变换
        x_layer2=torch.relu(x_layer2)        # 激活函数 ReLU：x<0 置 0，x>0 保留
        out =self.out(x_layer2)              # 输出层：线性变换，得到 2 个类别的原始分数
        out =torch.softmax(out,dim=-1)       # Softmax：把分数归一化成概率（dim=-1 表示对最后一个维度）
        return out                           # 返回最终概率输出

if __name__ == '__main__':
    my_model =model()                        # 实例化模型
    x = torch.randn(10,3)                    # 随机生成一个 batch：10 个样本，每个样本 3 个特征
    out =my_model(x)                         # 前向传播（等价于调用 my_model.forward(x)）
    print(out.shape)                         # 输出形状 [10,2]：10 个样本、每样本 2 个类别的概率
    # 注意：3 个类别的概率加和 = 1（Softmax 的性质）。

    summary(my_model,input_size=(3,),batch_size=8)
    # torchsummary：打印网络结构，显示每层输出形状与参数量，检查是否有 bug。

    for name,para in my_model.named_parameters():
        print(name)                          # 参数名字，如 layer1.weight / layer1.bias / ...
        print(para)                          # 参数张量（含 shape 与数值，requires_grad 为 True）
