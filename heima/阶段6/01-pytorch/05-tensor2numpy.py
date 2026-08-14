# ============================================================
# 文件主题：Tensor 转 numpy —— .numpy() 的「内存共享」陷阱
# 核心结论：
#   1. .numpy() 默认和原张量「共享内存」：改 numpy 会同步改 tensor（反之亦然）
#   2. 因此这里调用 .numpy().copy() 复制一份副本，切断共享关系
#   3. 本文件演示：对副本 data_numpy 修改后，原张量 data_tensor 不受影响
# 易混点：.numpy() 只对「CPU 上的张量」有效，GPU 张量要先 .cpu() 再转；
#         共享内存 vs 复制（.copy()）是"改了没生效"类 bug 最常见的来源
# ============================================================
import torch
import numpy as np

# manual_seed(2)：固定随机种子，保证下面 randint 的结果可复现
torch.random.manual_seed(2)
# torch.randint(0,10,[2,3])：生成 2 行 3 列、元素取值在 [0,10) 内的随机整数张量
# 三个参数分别表示：最小值 0、最大值 10（不含 10）、形状 [2,3]
data_tensor = torch.randint(0,10,[2,3])
# 查看变量类型，应输出 <class 'torch.Tensor'>（torch 张量）
print(type(data_tensor))

# .numpy()：把张量转成 numpy 数组。默认与张量共享内存；
# 加了 .copy() 之后得到独立副本，修改 data_numpy 就不会再影响 data_tensor
data_numpy=data_tensor.numpy().copy()
# 查看转换后的类型，应输出 <class 'numpy.ndarray'>
print(type(data_numpy))

# 修改 numpy 副本里第 0 行第 0 列的元素为 100
data_numpy[0][0]=100
# 打印 numpy 数组：第 0 行第 0 列已变成 100
print(data_numpy)
# 打印原张量：因为前面用了 .copy() 切断共享，这里 data_tensor 仍是原来的值，没有变成 100
print(data_tensor)