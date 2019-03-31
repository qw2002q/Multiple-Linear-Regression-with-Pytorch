import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(2, 1) # two in and one out
        #self.sigmoid = torch.nn.Sigmoid()
        #self.relu = torch.nn.ReLU()

    def weight_bias_reset(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                mean, std = 0, 0.5
                #print("before:",m.weight,m.bias)
                torch.nn.init.normal_(m.weight, mean, std)
                torch.nn.init.normal_(m.bias, mean, std)
                #print("after:",m.weight,m.bias)

    def forward(self, x):
        y_pred = self.l1(x)
        return y_pred
