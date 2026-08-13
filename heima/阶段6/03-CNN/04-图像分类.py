from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import DataLoader

# 数据获取
train_data = CIFAR10(root='data', train=True, transform=Compose([ToTensor()]))
test_data = CIFAR10(root='data', train=False, transform=Compose([ToTensor()]))


# print(test_data.data.shape)
# print(train_data.data.shape)
# print(train_data.classes)
# print(train_data.class_to_idx)
#
# plt.imshow(train_data.data[100])
# print(train_data.targets[100])
# plt.show()
# 模型构建
class imgClassification(nn.Module):
    # 初始化
    def __init__(self):
        super(imgClassification, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
        self.pooling1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.pooling2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer3 = nn.Linear(in_features=576, out_features=120)
        self.layer4 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, x):
        x = torch.nn.LeakyReLU(self.layer1(x))
        x = self.pooling1(x)
        x = torch.relu(self.layer2(x))
        x = self.pooling2(x)
        x = torch.reshape(x, [x.size(0), -1])
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        out = self.out(x)
        return out


model = imgClassification()


#
# summary(model,input_size=(3,32,32),batch_size=1)

# 模型训练

def train():
    # 损失函数
    cri = nn.CrossEntropyLoss()
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=[0.9, 0.99])
    # 遍历每个轮次
    epochs = 10
    loss_mean = []
    for epoch in range(epochs):
        dataloader= DataLoader(train_data,batch_size=2,shuffle=True)
        # 每个遍历batch
        loss_sum = 0
        sample = 0.1
        for x,y in dataloader:
            y_predict =model(x)
            # loss
            loss=cri(y_predict,y)
            loss_sum+=loss.item()
            sample+=1
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_mean.append(loss_sum/sample)
        print(loss_sum/sample)
    print(loss_mean)
    # 保存模型权重
    torch.save(model.state_dict(),'/Users/mac/Desktop/AI20深度学习/02-code/03-CNN/model.pth')

# train()
# 模型预测
def test():
    dataloader = DataLoader(test_data,batch_size=8,shuffle=False)
    # 加载模型
    model.load_state_dict(torch.load('/Users/mac/Desktop/AI20深度学习/02-code/03-CNN/model.pth'))

    # 遍历数据进行预测
    correct=0
    samples = 0
    for x,y in dataloader:
        y_predict = model(x)
        correct += (torch.argmax(y_predict,dim=-1)==y).sum()
        samples += len(y)
    acc = correct/(samples+0.000001)
    print(acc)

test()