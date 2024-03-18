import math
from random import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import SAGEConv

def stochastic_op(n):
    temp = math.floor(n)
    return temp +((n-temp) > random())

def qunt_op(X, qB):
    x = X.cpu().detach().numpy()
    record_min = []
    record_offset = []
    for dims in range(len(x)):
        _max = x[dims].max()
        _min = x[dims].min()
        _offset = _max - _min
        record_min.append(_min)
        record_offset.append(_offset)

        for value_in_dim in range(len(x[dims])):
            x[dims][value_in_dim] = qB*(x[dims][value_in_dim] - _min)/_offset
            x[dims][value_in_dim] = int(stochastic_op(x[dims][value_in_dim]))

    x = torch.from_numpy(x)

    return x.int(), record_min, record_offset

def dequnt_op(X, Mins, Offsets, qB):
    X1 = X.cpu().detach().numpy()
    for dims in range(len(X1)): 
        _min = Mins[dims]
        _offset = Offsets[dims]
        for value_in_dim in range(len(X1[dims])):
            X1[dims][value_in_dim] = _min + (_offset*X1[dims][value_in_dim])/qB
    # X2 = torch.tensor(X1)

    return X1