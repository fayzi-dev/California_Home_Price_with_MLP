import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import pandas as pd

data_train = pd.read_csv('train.csv')
x_train = data_train.drop('MedHouseVal', axis=1)
# print(x_train.head())
y_train = data_train['MedHouseVal']
# print(y_train.head())

data_test = pd.read_csv('test.csv')
x_test = data_test.drop('MedHouseVal', axis=1)
y_test = data_test['MedHouseVal']

x_train = torch.tensor(x_train.values, dtype=torch.float32)
print(x_train.shape)  # torch.Size([16512, 8])
y_train = torch.tensor(y_train.values, dtype=torch.float32)
print(y_train.shape)  # torch.Size([16512])

x_test = torch.FloatTensor(x_test.values)
print(x_test.shape)  # torch.Size([4128, 8])
y_test = torch.FloatTensor(y_test.values)
print(y_test.shape)  # torch.Size([4128])

# Normalize
mu = x_train.mean(dim=0)
std = x_train.std(dim=0)
x_train = (x_train - mu) / std
print(x_train.mean(dim=0))#tensor([ 1.6383e-07, -1.3862e-09, -5.4984e-08,  8.3169e-08, -1.7188e-07,
         # 1.4555e-08, -1.5668e-06,  1.3101e-06])
print(x_train.std(dim=0))
# tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])

x_test = (x_test - mu) / std
print(x_test.mean(dim=0))#tensor([ 0.0174,  0.0059,  0.0357,  0.0232, -0.0412, -0.0045,  0.0125, -0.0141])
print(x_test.std(dim=0))#tensor([1.0173, 0.9952, 1.3752, 1.2463, 0.8829, 0.7154, 0.9926, 0.9963])

#DataLoader
