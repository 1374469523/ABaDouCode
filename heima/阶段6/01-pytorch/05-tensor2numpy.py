import torch
import numpy as np

torch.random.manual_seed(2)
data_tensor = torch.randint(0,10,[2,3])
print(type(data_tensor))

data_numpy=data_tensor.numpy().copy()
print(type(data_numpy))

data_numpy[0][0]=100
print(data_numpy)
print(data_tensor)