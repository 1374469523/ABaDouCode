# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

# 实现注意力的计算
# 实现注意力的计算：要按照讲义上说明的注意力计算步骤

class MyAtten(nn.Module):
    def __init__(self, query_size, key_size, value_size1 , value_size2, output_size):
        super().__init__()
        # 定义属性
        self.query_size = query_size
        self.key_size = key_size
        self.value_size1 = value_size1
        self.value_size2 = value_size2
        self.output_size = output_size
        # 定义第一个全连接层作用：得到注意力计算的权重分数
        # 因为Q和K需要拼接才送入Linear层，因此该Linear层的输入维度：query_size+key_size
        # 该Linear输出维度是value_size1的原因是为了和value进行矩阵相乘
        self.atten = nn.Linear(self.query_size+self.key_size, value_size1)

        # 定义第二个全连接层作用：按照注意力计算的计算步骤的第三步，需要按照指定维度输出注意力结果，线形变换
        # 该Linear接受的输入，是Q和第一步计算的结果拼接后的张量
        self.linear = nn.Linear(self.query_size + self.value_size2, self.output_size)

    def forward(self, Q, K , V):
        # 1.按照注意力计算第一规则：Q和K先进行拼接,经过Linear层，再经过softmax得到权重分数
        # Q[0]--》[1, 32];K[0]--》[1, 32]-->cat之后[1, 64]；atten_weight代表权重分数:[1, 32]
        atten_weight = F.softmax(self.atten(torch.cat((Q[0], K[0]), dim=-1)), dim=-1)
        # 2.需要将第1步计算的atten_weight-->[1, 32]和V矩阵相乘：V--》[1, 32, 64]
        # temp--》[1, 1, 32] * [1, 32, 64]-->[1, 1, 64] # 第一步根据第一个注意力计算规则得到结果
        temp = torch.bmm(atten_weight.unsqueeze(dim=0), V)
        # 3.因为第一步有拼接操作，所以我们进行第二步：将Q和temp进行再次拼接
        # output是Q[0]-->[1, 32]和temp[0]-->[1, 64]拼接后的结果==》[1, 96]
        output = torch.cat((Q[0], temp[0]), dim=-1)
        # 4.我们需要根据计算步骤的第三步，对第3步拼接后的结果进行线形变化，因此需要linear
        # result-->[1, 1, 32]
        result = self.linear(output).unsqueeze(dim=0)

        return result, atten_weight



class MyAtten2(nn.Module):
    def __init__(self, query_size, key_size, value_size1, value_size2, output_size):
        super().__init__()
        # 定义属性
        self.query_size = query_size
        self.key_size = key_size
        self.value_size1 = value_size1
        self.value_size2 = value_size2
        self.output_size = output_size
        # 定义第一个全连接层作用：得到注意力计算的权重分数
        # 因为Q和K需要拼接才送入Linear层，因此该Linear层的输入维度：query_size+key_size
        # 该Linear输出维度是value_size1的原因是为了和value进行矩阵相乘
        self.atten = nn.Linear(self.query_size+self.key_size, value_size1)

        # 定义第二个全连接层作用：按照注意力计算的计算步骤的第三步，需要按照指定维度输出注意力结果，线形变换
        # 该Linear接受的输入，是Q和第一步计算的结果拼接后的张量
        self.linear = nn.Linear(self.query_size + self.value_size2, self.output_size)

    def forward(self, Q, K , V):
        # 1.按照注意力计算第一规则：Q和K先进行拼接,经过Linear层，再经过softmax得到权重分数
        # Q--》[1, 1, 32];K--》[1, 1, 32]-->cat之后[1, 1, 64]；atten_weight代表权重分数:[1, 1, 32]
        atten_weight = F.softmax(self.atten(torch.cat((Q, K), dim=-1)), dim=-1)
        # 2.需要将第1步计算的atten_weight-->[1, 32]和V矩阵相乘：V--》[1, 32, 64]
        # temp--》[1, 1, 32] * [1, 32, 64]-->[1, 1, 64] # 第一步根据第一个注意力计算规则得到结果
        temp = torch.bmm(atten_weight, V)
        # 3.因为第一步有拼接操作，所以我们进行第二步：将Q和temp进行再次拼接
        # output是Q-->[1, 1, 32]和temp-->[1,1, 64]拼接后的结果==》[1,1, 96]
        output = torch.cat((Q, temp), dim=-1)
        # 4.我们需要根据计算步骤的第三步，对第3步拼接后的结果进行线形变化，因此需要linear
        # result-->[1, 1, 32]
        result = self.linear(output)

        return result, atten_weight

if __name__ == '__main__':
    Q = torch.randn(1, 1, 32) # 32-->query_size
    K = torch.randn(1, 1, 32) # 32-->key_size
    V = torch.randn(1, 32, 64) # 32-->value_size1, 64-->value_size2

    # 实例化对象
    # my_attention = MyAtten(query_size=32, key_size=32, value_size1=32, value_size2=64, output_size=32)
    my_attention2 = MyAtten2(query_size=32, key_size=32, value_size1=32, value_size2=64, output_size=32)
    result, atten_weight =  my_attention2(Q, K, V)
    print(f'result--->{result.shape}')
    print(f'atten_weight--->{atten_weight.shape}')
