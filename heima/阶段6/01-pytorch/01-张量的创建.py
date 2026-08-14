# ============================================================
# 文件主题：张量的创建 —— 演示构造张量的几种常用方式
# 核心结论：
#   1. torch.Tensor(2,3) 传的是「形状」，只分配内存、不保证具体值，
#      打印出来可能是无意义的残留随机数（初学者最易踩坑的地方）
#   2. torch.Tensor([100]) 传的是「数据列表」，会按列表内容创建张量
#   3. torch.Tensor(numpy数组) 可把 numpy 数组直接转成 torch 张量
#   4. torch.IntTensor / torch.FloatTensor 是「指定 dtype 的类型构造函数」：
#      - IntTensor 会直接截断小数（10.5 -> 10）
#      - FloatTensor 得到 32 位浮点型（torch 默认 dtype）
#   易混点：torch.tensor（小写）是工厂函数，一定按「数据」创建；
#           torch.Tensor（大写）是类型构造器，传整数=形状，传列表/数组=数据
# ============================================================
import torch
import numpy as np

# 下面一组被注释掉的写法，展示「小写 torch.tensor」按数据创建张量
# torch.tensor(100) 表示传标量 100，创建 0 维（只有一个数）的张量
# print(torch.tensor(100))
# torch.tensor([10.5,2.3]) 表示传列表，创建一维张量 [10.5, 2.3]
# print(torch.tensor([10.5,2.3]))
# np.random.randn(10) 让 numpy 生成 10 个标准正态分布的随机数（一维数组，形状 (10,)）
# data = np.random.randn(10)
# print(data)
# torch.tensor(data) 把 numpy 数组转成 torch 张量
# print(torch.tensor(data))

# torch.Tensor(2, 3)：传两个整数 2 和 3，表示「形状 2 行 3 列」。
# 注意：这里只分配了内存，里面的值未初始化，打印出的数字是残留随机值、没有意义（也不一定是 0）
print(torch.Tensor(2, 3))
# torch.Tensor([100])：传「列表」时当作数据来用，创建一维张量 [100]
print(torch.Tensor([100]))
# torch.Tensor([10.5,2.3])：按列表数据 [10.5, 2.3] 创建一维张量（两个元素）
print(torch.Tensor([10.5,2.3]))
# np.random.randn(10)：numpy 生成 10 个标准正态分布的随机数，形状 (10,)，含 10 个元素
data = np.random.randn(10)
# 打印 numpy 数组，输出形如 [-0.37  0.95 ...] 的一行 10 个小数
print(data)
# torch.Tensor(data)：把 numpy 数组转成 torch 张量，dtype 变成 torch.float32（32 位浮点）
print(torch.Tensor(data))

# torch.IntTensor([10.5,2.3])：创建 int 类型张量，小数被直接截断 → 结果是 [10, 2]
print(torch.IntTensor([10.5,2.3]))
# torch.FloatTensor([100])：创建 float32（32 位浮点）类型张量 → 结果是 [100.]
print(torch.FloatTensor([100]))