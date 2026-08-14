# ============================================================
# 文件主题：完整线性回归小案例 —— 数据集 → DataLoader → 模型 → 训练 → 可视化
# 核心结论：
#   1. make_regression 生成带线性关系的数据，TensorDataset+DataLoader 打包成批
#   2. nn.Linear(1,1) 就是一个最简单的线性模型 y = w*x + b
#   3. 训练五步曲：预测(predict) → 算损失(loss) → 梯度清零(zero_grad)
#      → 反向传播(backward) → 更新参数(step)
#   4. 每个 epoch 遍历所有 batch 后记录平均损失，最后画「拟合直线」和「损失曲线」
# 训练直觉：loss.backward() 算出梯度后，optimizer.step() 让参数沿「负梯度」方向
#   走一小步（步长由学习率 lr 决定），重复多次后 w、b 收敛，直线拟合数据
# ============================================================
# 构造数据集
from sklearn.datasets import make_regression
# 构造适合torch数据集
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import torch

# 构建数据集
# make_regression：生成回归用的模拟数据（numpy 数组）
x, y, coef = make_regression(n_samples=100,  # 样本个数：一共生成 100 个样本
                             n_features=1,  # 特征维度：每个样本只有 1 个特征 x
                             noise=10,  # 噪声：叠加在 y 上的随机扰动（越大数据越散）
                             bias=1.5,  # 偏置：真实直线 y = coef*x + 1.5 中的截距
                             coef=True  # 返回,斜率：额外返回真实的斜率 coef（用于画真实直线）
                             )

# 散点图：把 100 个 (x,y) 样本画成散点，能看出大概的线性关系
plt.scatter(x, y)
# y_true = [coef*v+1.5 for v in x]   # 被注释：用真实斜率+截距算出的理想直线
# plt.plot(x,y_true)                 # 被注释：画理想直线（取消注释可与散点对比）
# plt.show()                         # 被注释：展示散点图（取消注释可先看一眼数据）
# plt.show()

# 数据获取

# 转换成tensor：sklearn 返回的是 numpy 数组，先转成 torch 张量才能喂给模型
x = torch.tensor(x)
y = torch.tensor(y)
# 构造适合torch数据集:100个数据
# TensorDataset(x, y)：把输入 x 和标签 y 打包成「数据集」，按下标即可拿到 (x, y) 对
dataset = TensorDataset(x, y)
# 构建batch数据
# DataLoader：按 batch 拆分数据。batch_size=8 表示每批 8 个样本（100 个分 13 批，
# 最后一批不足 8 个）；shuffle=True 每个 epoch 重新打乱顺序；
# drop_last=False 表示最后一批不够 8 个也保留（不丢弃）
daloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, drop_last=False)

# 构建模型:线性回归
# nn.Linear(in_features=1, out_features=1)：全连接线性层，内部有权重 w 和偏置 b
# 输入 1 维（x）、输出 1 维（y），即模型 y = w*x + b（w、b 会被随机初始化）
model = torch.nn.Linear(in_features=1,  # 输入x的维度
                        out_features=1  # 输出y的维度
                        )

# print(model.parameters())：打印参数对象。线性层含两个可学习参数：权重 (1,1) 和偏置 (1,)，
# 这里打印的是 Parameter 对象的信息；想看具体数值需打印 model.weight / model.bias
print(model.parameters())

# 模型训练
# 损失:均方误差
# MSELoss()：均方误差损失，衡量预测与真实之间「平均平方差距」，越小越好
cri = torch.nn.MSELoss()
# 优化器
# SGD(params=model.parameters(), lr=0.001)：随机梯度下降优化器
# params 传入要训练的参数；lr=0.001 是学习率，控制每次参数更新迈多大步子
#（太大容易发散、太小收敛慢）
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)
# 遍历"epoch batch
# loss_num：记录每个 epoch 的平均损失，方便最后画损失下降曲线
loss_num = []
# 遍历每个epoch：把整个数据集完整过一遍算一个 epoch，这里训练 100 轮
for i in range(100):
    sum = 0
    sample = 0
    # 获取batch数据：依次取每个 batch，x_ 是批输入(8,1)、y_ 是批标签(8,)
    for x_, y_ in daloader:
        # 模型预测
        # model(x_.type(torch.float32))：x_ 默认是 float64，先转成 float32 再前向传播
        # 得到预测值 y_predict，形状 (8,1)
        y_predict = model(x_.type(torch.float32))
        # 损失计算
        # y_.reshape(-1, 1)：把标签从形状 (8,) 变成 (8,1)，与预测对齐；再转 float32
        loss = cri(y_predict, y_.reshape(-1, 1).type(torch.float32))
        sum += loss.item()   # .item() 取出 loss 标量张量里的数值，用于累计
        sample += len(y_)    # 累计本 batch 的样本数，用于算平均损失
        # 梯度清零
        # zero_grad()：把上一次 backward 留下的梯度清零
        #（PyTorch 的梯度默认「累加」，不清零会导致本次梯度叠加出错）
        optimizer.zero_grad()
        # 自动微分
        # backward()：反向传播，自动算出 loss 对 w、b 的梯度，存入各自 .grad
        loss.backward()
        # 更新参数
        # step()：按 SGD 规则用梯度更新模型参数 w、b（沿负梯度方向走一小步）
        optimizer.step()
    loss_num.append(sum / sample)  # 记录这个 epoch 的平均损失

# 绘制拟合直线
# linspace：从 x 的最小值到最大值均匀取 1000 个点，用于画光滑的直线
x = torch.linspace(x.min(), x.max(), 1000)
# 用训练好的模型参数计算预测直线上的点：v * model.weight + model.bias
# model.weight / model.bias 是训练后收敛的权重和偏置
y1 = torch.tensor([v * model.weight + model.bias for v in x])
# 用真实斜率 coef 和真实截距 1.5 计算真实直线（作为对照基准）
y2 = torch.tensor([v * coef + 1.5 for v in x])
plt.plot(x, y1, label='train')  # 蓝色曲线：训练得到的拟合直线
plt.plot(x, y2, label='real')   # 橙色曲线：真实直线，训练直线应逼近它
plt.grid()                      # 加网格方便读数
plt.legend()                    # 显示图例 label
plt.show()                      # 弹出第一张图：拟合直线 vs 真实直线

# 绘制损失变化曲线
# 横轴是 epoch（0-99），纵轴是每个 epoch 的平均损失 loss_num
# 正常情况曲线应单调下降并趋于平稳，说明模型在收敛
plt.plot(range(100), loss_num)
plt.grid()
plt.show()  # 弹出第二张图：损失曲线
