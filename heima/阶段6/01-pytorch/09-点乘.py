# ============================================================
# 文件主题：点乘（逐元素乘）—— torch.mul 与 * 的关系
# 核心结论：
#   1. torch.mul(a,b) 和 a * b 完全等价，都是「逐元素相乘」：对应位置两个数相乘
#   2. 逐元素乘要求两个张量形状能「广播」：从最右边维度往左看，
#      每一维要么相等、要么其中一个是 1
#   3. 本示例中 data1 是 (3,3)、data2 是 (2,3)：
#      最右维都是 3 没问题，但倒数第 2 维是 3 vs 2，既不等、也都不是 1
#      → 无法广播，直接运行到这里会报错（这是有意的对比演示）
# 易混点：逐元素乘（* / mul）≠ 矩阵乘（@ / matmul），
#         矩阵乘要求内维相等（见下一个文件 10-矩阵乘法）
# ============================================================
import torch

# manual_seed(22)：固定随机种子，让 data1 可复现
torch.random.manual_seed(22)
# torch.randint(0,10,[3,3])：生成 3 行 3 列的随机整数张量
data1 = torch.randint(0,10,[3,3])
print(data1)

# 换一个新种子，生成另一组随机数
torch.random.manual_seed(23)
# torch.randint(0,10,[2,3])：生成 2 行 3 列的随机整数张量
data2 = torch.randint(0,10,[2,3])
print(data2)

# torch.mul(data1, data2)：逐元素相乘。但 (3,3) 与 (2,3) 形状不满足广播条件，
# 运行到这一行会报错，例如 "The size of tensor a (3) must match ..."
# 借此体会：为什么逐元素乘要求形状能广播
print(torch.mul(data1, data2))

# data1 * data2：与 torch.mul 完全等价，同样是逐元素乘、同样的广播规则 → 同样报错
print(data1 * data2)