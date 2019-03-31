import numpy as np
import torch
import torch.nn as nn

def ReadData(file): #Read Train/Test Data
    if file == "train":
        data = np.loadtxt('./data/dataForTraining.txt', dtype=np.float32)
    elif file == "test":
        data = np.loadtxt('./data/dataForTesting.txt', dtype=np.float32)
    elif file == "debug":
        data = np.loadtxt('./data/debug.txt', dtype=np.float32)
    else:
        print("ERROR!!: invalid parameter, please input train/test")
        return

    x = torch.from_numpy(data[:,0:-1])
    y = torch.from_numpy(data[:,[-1]])
    return x,y
