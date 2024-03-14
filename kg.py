from torch_geometric.datasets import Planetoid
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import SAGEConv
import math
from random import random
from datetime import datetime
import os
import numpy as np
from memory_profiler import profile
dataset = Planetoid(root='../../dataset/', name='Cora')
B = 31.


# print(dataset[0])
'''
Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])
類型:7
圖數量:1

2708個點，1433維
10556條有向邊，5278條邊

train資料集的點數量:140
test資料集的點數量 :999
valid資料集的點數量:500
value 範圍:-1~1
'''
def sto_op(x):
    a = math.floor(x)
    return a +((x-a) > random())

def qunt(x):
    # 回傳值:修改後的int tensor, 這個tensor的offset, min值
    # 關於stochastically rounding 寫法https://stackoverflow.com/questions/62336144/stochastically-rounding-a-float-to-an-integer
    # 居然是O(1) operation...
    # i=0
    # for dim in x:
    #     _Max = dim.max()
    #     _Min = dim.min()
    #     offset = _Max - _Min
    #     #print(_Max+_Min)
    #     dim = B*(dim-(_Min))/offset
    #     # dim = dim/offset
    #     # x[i] = dim
    #     i+=1
    # detach阻斷反向傳播
    x1 = x.cpu().detach().numpy()
    record_Min = []
    record_Min.clear()
    record_offset = []
    record_offset.clear()
    for dims in range(len(x1)):     #2708
        _Max = x1[dims].max()
        _Min = x1[dims].min()
        _offset = _Max - _Min
        record_Min.append(_Min)
        record_offset.append(_offset)
        # x[i] = B*(x[i]-_Min)/_offset
        #print(len(x[dims]))
        #print(_Min, _Max, _offset)
        for value_in_dim in range(len(x1[dims])):       #32
            #print(x[dims][value_in_dim])
            x1[dims][value_in_dim] = B*(x1[dims][value_in_dim] - _Min)/_offset
            if not math.isnan(x1[dims][value_in_dim]):
                x1[dims][value_in_dim] = sto_op(x1[dims][value_in_dim])
    '''暴力int收斂'''
    # for dim in range(len(x)):
    #     x[dim] = x[dim].int()
    x1 = torch.from_numpy(x1)
    # print(x.int().type())
    return x1, record_Min, record_offset

def dequnt(x, Mins, Offsets):
    x1 = x.cpu().detach().numpy()
    for dims in range(len(x1)):  #2708
        _min = Mins[dims]
        _offset = Offsets[dims]
        for value_in_dim in range(len(x1[dims])):
            x1[dims][value_in_dim] = _min + (_offset*x1[dims][value_in_dim])/B
    
    return torch.tensor(x1, requires_grad=True)


class GraphSAGE(nn.Module):
    def __init__(self):
        super(GraphSAGE, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(1433, 64))
        self.convs.append(SAGEConv(64, 32))

        # self.int_conv1 = []
        # self.record_Min_1 = []
        # self.record_offset_1 = []
        # self.int_conv2 = []
        # self.record_Min_2 = []
        # self.record_offset_2 = []
        
        self.dropout = 0.5

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # print("parameter :\n", self.convs.parameters())
        
        
        # for parm in self.convs[0].parameters():
        #     print("conv1 parm is :\n", parm)
        #     print("conv1 size:\n", parm.size())
        # print("\n\n")

        # x = F.relu(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # print("after tiny: \n", qunt(x))
        x = self.convs[0](x, edge_index)
        # print("x :\n", x)
        # for parm in self.convs[0].parameters():
        #     print("parm is :\n", parm)
        #     print("parm size:\n", parm.size())
        x = self.convs[1](x, edge_index)
        # x = x.bfloat16()
        # print("\nx's type", x.type())
        x1, record_min, record_offset = qunt(x)
        # print("\nafter qunt ", x1)
        x2 = dequnt(x1, record_min, record_offset)
        # print("\nafter dequnt, ", x2)
        print(x2)
        x = F.relu(x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

# device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
device = torch.device(device)
model = GraphSAGE()
# model.cuda()
filter_fn = filter(lambda p: p.requires_grad, model.parameters())
learning_rate = 1e-2
weight_decay = 5e-3
epochs = 100
opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

data = dataset[0]
print("data x's size :\n", data.x.size())
data.to(device)
losses, val_accs = [], []
loss_fn = nn.NLLLoss()

# @profile
def train():
    start =  datetime.now()
    for epoch in range(epochs):
        print("\n================\n")
        # print(epoch)
        model.train()
        opt.zero_grad()
        out = model(data)       # 這邊call forward
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        # print("\n\nafter forward\n\n")
        # print("loss is : \n", loss)
        # print("loss is :\n")
        # for single_loss in loss.item():
        #     print(single_loss)
        loss.backward()
        opt.step()
        # for par in model.convs[0].parameters():
        #     print("\nconv1 is :\n", par)
        #     print("\nconv1 size:\n", par.size())
        losses.append(loss.item())
        if epoch % 1 == 0:
            model.eval()  

            with torch.no_grad():
                # max(dim=1) returns (values, indices) tuple; only need indices
                pred = model(data).max(dim=1)[1]
                correct = float(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
                acc = correct / data.val_mask.sum().item()

            val_accs.append(acc)
            print('Epoch: {:03d}, Loss: {:.5f}, Val Acc.: {:.3f}'.format(epoch, loss.item(), acc))
            
        else:
            val_accs.append(val_accs[-1])
    end = datetime.now()
    print("total time consume : ", end - start)
train()

print("Maximum accuracy: {0}".format(max(val_accs)))
print("Minimum loss: {0}".format(min(losses)))

# plt.title(dataset.name)
# plt.plot(losses, label="training loss" + " - " + "GraphSAGE")
# plt.plot(val_accs, label="val accuracy" + " - " + "GraphSAGE")
# plt.legend()
# plt.show()


