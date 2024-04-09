import math
from random import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import SAGEConv
import copy
from datetime import datetime
def stochastic_op(n):
    temp = math.floor(n)
    return temp +((n-temp) > random())

def qunt_op(X : torch.tensor, qB : int):
    # x = X.cpu().detach().numpy()
    x = X.clone().detach()

    record_min = []
    record_offset = []
    start = datetime.now()
    for dims in range(len(x)):
        _max = x[dims].max()
        _min = x[dims].min()
        _offset = _max - _min
        record_min.append(_min)
        record_offset.append(_offset)

        for value_in_dim in range(len(x[dims])):
            x[dims][value_in_dim] = qB*(x[dims][value_in_dim] - _min)/_offset
            # x[dims][value_in_dim] = int(stochastic_op(x[dims][value_in_dim]))
    end = datetime.now()
    # print(end - start)
    # x = torch.from_numpy(x)
    # print(x)
    return x.to(torch.uint8), record_min, record_offset

def dequnt_op(X : torch.tensor, Mins, Offsets, qB):
    # x = X.clone().detach()
    x = X.to(torch.float32)
    for dims in range(len(x)): 
        _min = Mins[dims]
        _offset = Offsets[dims]
        for value_in_dim in range(len(x[dims])):
            x[dims][value_in_dim] = _min + (_offset*x[dims][value_in_dim])/qB
    # X2 = torch.tensor(X1)
    return x