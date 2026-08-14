# ============================================================
# 文件主题：形状操作 —— cat（拼接）、reshape/view、unsqueeze/squeeze、
#           transpose/permute、is_contiguous/contiguous
# 核心结论：
#   1. torch.cat([a,b], dim=2)：沿第 2 维拼接两个张量。
#      要求「除拼接维度外」其余维度完全一致：data1(4,5,3) 与 data2(4,5,5)
#      → 在 dim=2 上拼接，3+5=8，结果形状 (4,5,8)
#   2. 下面被注释的代码依次演示：size（取某维长度）、reshape（改形状）、
#      unsqueeze/squeeze（增/删长度为 1 的维度）、transpose（交换两个维度）、
#      permute（按列表重排所有维度）、view（展平/改形状，要求内存连续）
# 易混点：view 要求内存连续（不连续要先 .contiguous()），reshape 更灵活；
#         transpose/permute 只是「换维度顺序」，底层内存没搬，可能不再连续
# ============================================================
import torch

# manual_seed(22)：固定随机种子，让 data1 可复现
torch.random.manual_seed(22)
# torch.randint(0,10,[4,5,3])：生成形状为 (4,5,3) 的随机整数张量（4×5×3=60 个元素）
data1 =torch.randint(0,10,[4,5,3])
print(data1)

# manual_seed(23)：换种子，生成另一组随机数
torch.random.manual_seed(23)
# torch.randint(0,10,[4,5,5])：生成形状为 (4,5,5) 的随机整数张量
data2 =torch.randint(0,10,[4,5,5])
print(data2)
# data.size(2)：返回第 2 维的长度（data1 的第 2 维是 3，形状 (4,5,3)）
# 注意：这里的 data 变量在下面并未定义，仅是老师演示的写法，取消注释运行会报错
# print(data.size(2))
# data.reshape(-1)：-1 表示「由 PyTorch 自动推算」，把张量展平成一维；
# .shape 查看结果形状（若按 data1 计算，展平后是 60 个元素）
# print(data.reshape(-1).shape)

# unsqueeze(dim=1)：在第 1 维位置插入一个长度为 1 的维度（加一个"括号"）：
# (4,5,3) → (4,1,5,3)；再 unsqueeze(dim=-1) 在末尾再加一维 → (4,1,5,3,1)
# data1 =data.unsqueeze(dim=1).unsqueeze(dim=-1)
# print(data1.shape)
# squeeze()：不加参数时把所有长度为 1 的维度全部去掉：(4,1,5,3,1) → (4,5,3)
# print(data1.squeeze().shape)

# 目标是[3,4,5,2]：下面演示通过三次 transpose，把维度顺序换成 (3,4,5,2)
# 一次 transpose 只能交换两个维度，所以需要多步
# data1 = torch.transpose(data, 0, 2)   # 交换第 0 和第 2 维
# data2 = torch.transpose(data1, 1, 2)  # 再交换第 1 和第 2 维
# data3 = torch.transpose(data2, 2, 3)  # 再交换第 2 和第 3 维
# print(data3.shape)
# torch.permute(data, [2, 0, 3, 1])：用下标列表一次性重排所有维度，等价于上面三步
# print(torch.permute(data, [2, 0, 3, 1]).shape)
# data.permute([2, 0, 3, 1])：permute 也有张量方法写法，结果一样
# print(data.permute([2, 0, 3, 1]).shape)
# data.is_contiguous()：判断张量底层内存是否连续。
# transpose/permute 换维度后通常不再连续（返回 False）
# print(data.is_contiguous())
# data.view(-1)：view 要求内存连续，data 若不连续会报错；连续时可用 -1 展平
# print(data.view(-1).shape)
# data1 = torch.transpose(data,0,1)   # 交换第 0、1 维后大概率不连续
# print(data1.is_contiguous())
# data2=data1.contiguous()   # contiguous()：把数据按新维度顺序重排成连续内存
# print(data2.view(-1).shape)  # 连续之后再 view 就没问题了
#
# if data.is_contiguous():    # 更稳健的写法：先判断内存是否连续
#     data.view(-1)           # 连续就直接 view
# else:
#     data.contiguous().view(-1)   # 不连续就先 contiguous() 再 view

# torch.cat([data1, data2], dim=2)：沿第 2 维拼接。
# data1(4,5,3) 与 data2(4,5,5) 除第 2 维外都是 (4,5)，满足拼接条件
# 第 2 维上 3+5=8，所以结果形状是 (4,5,8)
print(torch.cat([data1, data2], dim=2).shape)