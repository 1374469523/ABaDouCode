# ============================================================
# 文件主题：dtype（张量数据类型）与类型转换
# 核心结论：
#   1. dtype 决定每个元素占多少内存以及如何解释数据（如 int32、float32 各占 4 字节）
#   2. torch.randn 默认生成 float32（32 位浮点）类型的张量
#   3. 两种类型转换写法等价：data.type(torch.IntTensor) 和 data.int()，
#      都是把 float 转成 int；转换时小数被直接截断（10.9 -> 10，不是四舍五入）
# 易混点：.type(torch.IntTensor) 里传的是「张量类型类」torch.IntTensor，
#         和 .int() 是同一回事，两种写法都要会认
# ============================================================
import torch

# torch.randn(2,3)：生成 2 行 3 列、标准正态分布的随机张量，默认 dtype 是 float32
data =torch.randn(2,3)
# data.dtype：查看张量的数据类型，这里应输出 torch.float32
print(data.dtype)
# data.type(torch.IntTensor)：把 data 转成 int 类型的新张量，再取 .dtype 查看
# → 输出 torch.int32。注意 float 转 int 是截断小数，不是四舍五入
print(data.type(torch.IntTensor).dtype)
# data.int()：等价于上面的 .type(torch.IntTensor)，是更简洁的写法 → 输出 torch.int32
print(data.int().dtype)