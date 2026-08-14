# ============================================================
# 文件主题：指定值张量 —— ones（全 1）、zeros（全 0）、full（填固定值）
# 核心结论：
#   1. ones / zeros 传「形状」生成指定形状的全 1 / 全 0 张量
#   2. 带 _like 后缀的版本（ones_like 等）不传形状，而是「模仿另一个张量的形状」
#   3. full 需要同时给出「形状」和「填充值」两个参数
# 易混点：带 _like 的版本参数是一个「张量」，而不是整数形状，别写反
# ============================================================
import torch

# 先创建一个 2 行 3 列、标准正态分布随机数的张量，作为后续 _like 参照形状
data = torch.randn(2,3)
# torch.ones(4, 5)：生成 4 行 5 列的全 1 张量（默认 dtype 为 float32）
print(torch.ones(4, 5))
# torch.ones_like(data)：生成和 data 形状一样（2 行 3 列）的全 1 张量
print(torch.ones_like(data))
# torch.zeros(4, 5)：生成 4 行 5 列的全 0 张量
print(torch.zeros(4, 5))
# torch.zeros_like(data)：生成和 data 形状一样（2 行 3 列）的全 0 张量
print(torch.zeros_like(data))
# torch.full([4, 5], 100)：第一个参数传「形状列表」[4,5]，第二个参数 100 是填充值
# → 生成 4 行 5 列、所有元素都是 100 的张量
print(torch.full([4, 5], 100))
# torch.full_like(data, 200)：生成和 data 形状一样（2 行 3 列）、所有元素都是 200 的张量
print(torch.full_like(data, 200))