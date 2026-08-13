import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torchsummary import summary

# 1.获取数据
# 1.1 读取数据
data = pd.read_csv('/Users/mac/Desktop/AI20深度学习/02-code/02-神经网络/data/手机价格预测.csv')
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 1.2 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# print(x_train)
x_train = torch.tensor(x_train.values, dtype=torch.float32)
x_test = torch.tensor(x_test.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.int64)
y_test = torch.tensor(y_test.values, dtype=torch.int64)
# 1.3 封装tensor
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

# 1.4 构建数据迭代器
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)


# 2.模型构建
# 类
class model(nn.Module):
    # init
    def __init__(self):
        super(model, self).__init__()
        self.layer1 = nn.Linear(in_features=20, out_features=64)
        self.layer2 = nn.Linear(in_features=64, out_features=128)
        self.layer3 = nn.Linear(in_features=128, out_features=4)
        self.dropout = nn.Dropout(p=0.9)

    # forward
    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = torch.relu(x)
        out = self.layer3(x)
        return out


# 3.模型训练
def train():
    phone_model = model()
    # 损失
    cri = nn.CrossEntropyLoss()
    # 优化器
    optimizer =torch.optim.SGD(phone_model.parameters(),lr=0.01)
    # 遍历
    eopches = 20
    for epoch in range(eopches):
        loss_sum=0
        sample=0.1
        for x,y in train_dataloader:
            y_predict =phone_model(x)
            loss =cri(y_predict,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum+=loss.item()
            sample+=1
        print(loss_sum/sample)
    torch.save(phone_model.state_dict(),'/Users/mac/Desktop/AI20深度学习/02-code/02-神经网络/data/myphone.pth')


# 4.模型预测
def test():
    my_model=model()
    my_model.load_state_dict(torch.load('/Users/mac/Desktop/AI20深度学习/02-code/02-神经网络/data/myphone.pth'))

    correct=0
    for x,y in test_dataloader:
        y_predict = my_model(x)
        y_index =torch.argmax(y_predict,dim=1)
        correct+=(y_index==y).sum()
    acc =correct.item()/len(test_dataset)
    print(acc)




if __name__ == '__main__':
    my_model = model()
    summary(my_model, input_size=(20,), batch_size=10)
    # train()
    test()
