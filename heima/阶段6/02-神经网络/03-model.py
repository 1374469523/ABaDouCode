import torch
import torch.nn as nn
from torchsummary import summary

# 类
class model(nn.Module):
    # init:定义层
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=3,out_features=3)
        nn.init.kaiming_normal_(self.layer1.bias)
        self.layer2 = nn.Linear(in_features=3,out_features=2)
        nn.init.xavier_uniform_(self.layer2.weight)
        self.out = nn.Linear(in_features=2,out_features=2)
        nn.init.uniform_(self.out.weight)

    # forward:前向传播
    def forward(self,x):
        x_layer1 = self.layer1(x)
        x_layer1=torch.sigmoid(x_layer1)
        x_layer2 =self.layer2(x_layer1)
        x_layer2=torch.relu(x_layer2)
        out =self.out(x_layer2)
        out =torch.softmax(out,dim=-1)
        return out

if __name__ == '__main__':
    my_model =model()
    x = torch.randn(10,3)
    out =my_model(x)
    print(out.shape)

    summary(my_model,input_size=(3,),batch_size=8)

    for name,para in my_model.named_parameters():
        print(name)
        print(para)