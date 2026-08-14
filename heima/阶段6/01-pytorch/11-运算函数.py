# ============================================================
# 文件主题：常用运算函数 —— mean / sum（聚合）与 sqrt / pow / exp / log（数学函数）
# 核心结论：
#   1. 不传 dim：对整张所有元素求均值 / 求和，返回一个标量（0 维张量）
#   2. 传 dim 参数：沿着某个「轴」聚合，dim 指定的那个维度会被压缩掉
#      - 对 (3,4) 张量：mean(dim=0) → 沿行方向聚合，结果形状 (4,)
#      - mean(dim=1) → 沿列方向聚合，结果形状 (3,)
#   3. sqrt/pow/exp/log 都是「逐元素」的数学运算，不改变张量形状
# 易混点：dim=0 是「跨第一维（行）」聚合，dim=1 是「跨第二维（列）」聚合，
#         很容易搞反方向，记住「dim 指定的是要被压缩掉的那个维度」
# ============================================================
import torch

# manual_seed(22)：固定随机种子，让随机结果可复现
torch.random.manual_seed(22)
# torch.randn([3,4])：生成 3 行 4 列、标准正态分布（均值 0、方差 1）的随机张量
data = torch.randn([3,4])
# 打印原始张量：3 行 4 列的浮点矩阵
print(data)

# data.mean()：对全部 3*4=12 个元素求平均，返回一个标量（0 维张量）
print(data.mean())
# data.mean(dim=0)：沿第 0 维（行方向）聚合，即对每一列求平均
# → 返回 4 个值，形状 (4,)，每个值对应一列的平均
print(data.mean(dim=0))
# data.mean(dim=1)：沿第 1 维（列方向）聚合，即对每一行求平均
# → 返回 3 个值，形状 (3,)，每个值对应一行的平均
print(data.mean(dim=1))

# data.sum()：对全部元素求和，返回一个标量
print(data.sum())
# data.sum(dim=0)：沿行方向聚合，对每一列求和 → 形状 (4,)
print(data.sum(dim=0))
# data.sum(dim=1)：沿列方向聚合，对每一行求和 → 形状 (3,)
print(data.sum(dim=1))

# data.sqrt()：对每个元素开平方（逐元素运算）。含负数时会得到 NaN（非数）
print(data.sqrt())
# torch.pow(data, 2)：每个元素取 2 次方（幂），等价于 data ** 2
print(torch.pow(data, 2))
# torch.pow(2, data)：底数是 2、指数是 data 里的每个数，即逐元素计算 2^x
print(torch.pow(2, data))
# data.exp()：每个元素求自然指数 e^x（e≈2.718）
print(data.exp())

# data.log()：每个元素求自然对数（以 e 为底）。含负数时得到 NaN
print(data.log())
# data.log2()：每个元素求以 2 为底的对数
print(data.log2())
# data.log10()：每个元素求以 10 为底的对数
print(data.log10())