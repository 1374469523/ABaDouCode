# ============================================================
# 文件主题：矩阵乘法 —— @ 与 torch.matmul
# 核心结论：
#   1. data1 @ data2 和 torch.matmul(data1, data2) 完全等价，都是矩阵乘法
#   2. 矩阵乘法要求「内维相等」：形状 (m,k) 的矩阵 @ (k,n) 的矩阵 → 结果 (m,n)
#      （A 的列数必须等于 B 的行数，否则报错）
#   3. 本示例：(3,4) @ (4,5) → 内维都是 4，合法，结果形状为 (3,5)
# 通俗理解：结果第 (i,j) 个元素 = A 的第 i 行 与 B 的第 j 列 对应相乘再求和
# 易混点：矩阵乘 @（要求内维相等）vs 逐元素乘 *（要求形状广播），规则完全不同
# ============================================================
import torch

# manual_seed(22)：固定随机种子，让 data1 可复现
torch.random.manual_seed(22)
# torch.randint(0,10,[3,4])：生成 3 行 4 列的随机整数张量，即形状 (3,4)
data1 = torch.randint(0,10,[3,4])
print(data1)

# manual_seed(23)：换种子，生成另一组随机数
torch.random.manual_seed(23)
# torch.randint(0,10,[4,5])：生成 4 行 5 列的随机整数张量，即形状 (4,5)
data2 = torch.randint(0,10,[4,5])
print(data2)

# data1 @ data2：矩阵乘法。(3,4)@(4,5)：A 的列数 4 == B 的行数 4，内维相等才合法，
# 结果形状 = (A 的行数, B 的列数) = (3,5)
print(data1 @ data2)
# torch.matmul(data1, data2)：功能和 @ 完全一样，是更正式的函数写法，结果同为 (3,5)
print(torch.matmul(data1, data2))
