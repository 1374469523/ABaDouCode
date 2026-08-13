# 构造数据集
from sklearn.datasets import make_regression
# 构造适合torch数据集
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import torch

# 构建数据集
x, y, coef = make_regression(n_samples=100,  # 样本个数
                             n_features=1,  # 特征维度
                             noise=10,  # 噪声
                             bias=1.5,  # 偏置
                             coef=True  # 返回,斜率
                             )

plt.scatter(x, y)
# y_true = [coef*v+1.5 for v in x]
# plt.plot(x,y_true)
# plt.show()

# 数据获取

# 转换成tensor
x = torch.tensor(x)
y = torch.tensor(y)
# 构造适合torch数据集:100个数据
dataset = TensorDataset(x, y)
# 构建batch数据
daloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, drop_last=False)

# 构建模型:线性回归
model = torch.nn.Linear(in_features=1,  # 输入x的维度
                        out_features=1  # 输出y的维度
                        )

print(model.parameters())

# 模型训练
# 损失:均方误差
cri = torch.nn.MSELoss()
# 优化器
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)
# 遍历"epoch batch
loss_num = []
# 遍历每个epoch
for i in range(100):
    sum = 0
    sample = 0
    # 获取batch数据
    for x_, y_ in daloader:
        # 模型预测
        y_predict = model(x_.type(torch.float32))
        # 损失计算
        loss = cri(y_predict, y_.reshape(-1, 1).type(torch.float32))
        sum += loss.item()
        sample += len(y_)
        # 梯度清零
        optimizer.zero_grad()
        # 自动微分
        loss.backward()
        # 更新参数
        optimizer.step()
    loss_num.append(sum / sample)

# 绘制拟合直线
x = torch.linspace(x.min(), x.max(), 1000)
y1 = torch.tensor([v * model.weight + model.bias for v in x])
y2 = torch.tensor([v * coef + 1.5 for v in x])
plt.plot(x, y1, label='train')
plt.plot(x, y2, label='real')
plt.grid()
plt.legend()
plt.show()

# 绘制损失变化曲线
plt.plot(range(100), loss_num)
plt.grid()
plt.show()
