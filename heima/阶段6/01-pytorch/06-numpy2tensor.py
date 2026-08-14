# ============================================================
# 文件主题：numpy 转 Tensor —— from_numpy（共享）vs Tensor()（复制）
# 核心结论：
#   1. torch.from_numpy(数组)：与 numpy 数组「共享内存」，改一个另一个也会变
#      （本文件把它注释掉了，正是为了和下面 Tensor() 的「复制」行为做对比）
#   2. torch.Tensor(numpy数组)：会「复制一份」数据，两者不再共享内存
#   3. 因此下面修改 data_tensor 后，data_numpy 保持不变
# 易混点：from_numpy 是共享、Tensor() 是复制，这是"改了没生效/不该生效"最经典的坑
# ============================================================
import torch
import numpy as np

# numpy 创建一维数组 [1, 2, 3]
data_numpy = np.array([1,2,3])
# 被注释掉的写法：torch.from_numpy(data_numpy.copy())
# from_numpy 会与 numpy 数组共享内存；这里先 copy() 造独立副本再转，以避免共享
# data_tensor=torch.from_numpy(data_numpy.copy())
# torch.Tensor(data_numpy)：把 numpy 数组「复制一份」转成张量，此后二者互不影响
data_tensor=torch.Tensor(data_numpy)
# 修改张量的第 0 个元素为 10
data_tensor[0] = 10
# 打印 numpy 数组：仍是 [1 2 3]，未被修改（证明 Tensor() 是复制而非共享）
print(data_numpy)
# 打印张量：已是 [10. 2. 3.]，只有张量自己被修改
print(data_tensor)