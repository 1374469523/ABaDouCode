# ============================================================
# 文件主题：基本运算 —— add / sub / mul / div / neg（逐元素运算 + 标量广播）
# 核心结论：
#   1. add/sub/mul/div/neg 都是「逐元素」运算：张量里的每个数分别与标量计算
#   2. 张量 + 标量 时，标量会自动「广播」扩展到张量的每个位置
#   3. 这些方法默认返回「新张量」，不会修改原 data（想原地修改要加下划线，如 add_）
# 易混点：div 是真除法，整数张量做 div 会得到浮点结果，dtype 可能发生变化
# ============================================================
import torch

# manual_seed(22)：固定随机种子，保证结果可复现
torch.random.manual_seed(22)
# torch.randint(0,10,[2,3])：生成 2 行 3 列、元素取值在 [0,10) 的随机整数张量
data = torch.randint(0,10,[2,3])
# 打印原始张量：2 行 3 列的整数矩阵
print(data)

# data.add(10)：每个元素都加 10（逐元素 + 标量广播），返回新张量，原 data 不变
print(data.add(10))
# data.sub(10)：每个元素都减 10
print(data.sub(10))
# data.mul(10)：每个元素都乘 10
print(data.mul(10))
# data.div(10)：每个元素都除以 10（真除法，整型结果会变成浮点型）
print(data.div(10))
# data.neg()：取负，每个元素变成相反数
print(data.neg())
# 被注释掉的 print(data)：取消注释可以看到，以上操作都没有改变原张量 data
# 原因：add/sub/mul/div/neg 都是「非原地」操作，返回的是新张量
# print(data)