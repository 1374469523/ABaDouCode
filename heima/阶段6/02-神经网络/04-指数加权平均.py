import torch
import matplotlib.pyplot as plt
# 创建数据
t = torch.randint(0,40,[30])
print(t)
days = torch.arange(0,30,1)
plt.plot(days,t)
plt.scatter(days,t)
plt.show()

# 指数加权平均
t_avg = []
beta = 0.9
for i,temp in enumerate(t):
    if i == 0:
        t_avg.append(temp)
        continue
    # 公式
    s =beta*t_avg[i-1]+(1-beta)*temp
    t_avg.append(s)

plt.plot(days,t_avg)
plt.scatter(days,t)
plt.show()


