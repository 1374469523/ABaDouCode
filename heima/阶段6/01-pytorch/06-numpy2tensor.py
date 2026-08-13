import torch
import numpy as np

data_numpy = np.array([1,2,3])
# data_tensor=torch.from_numpy(data_numpy.copy())
data_tensor=torch.Tensor(data_numpy)
data_tensor[0] = 10
print(data_numpy)
print(data_tensor)