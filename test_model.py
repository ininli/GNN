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
import matplotlib.pyplot as plt
from quntize_func import qunt_op, dequnt_op

dataset = Planetoid(root='../../dataset/', name='Cora')
B = 64


class GraphSAGE(nn.Module):
    def __init__(self):
        super(GraphSAGE, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(1433, 32))
        self.convs.append(SAGEConv(32, 32))
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.dropout = 0.5
        self.post_mp = nn.Sequential(
            nn.Linear(32, 32),
            nn.Dropout(self.dropout),
            nn.Linear(32, 7)
        )

    def forward(self, data):
        # if model.training != True:
        #     start = datetime.now()
        #             # x1, temp_min, temp_offset = qunt_op(x, B)
        #             # cent = datetime.now()
        #             # # print("qunt time :", cent - start)
        #             # x2 = dequnt_op(x1, temp_min, temp_offset, B)
        #             # x1 = torch.quantize_per_tensor(x, 0.1, 0, torch.quint8)
        #     x2 = torch.dequantize(x1)

        #     # print("default", x)
        #     # print("quant ", x1)
        #     # print("dequnt", x2)
        #     end = datetime.now()
        #     # print("a qunt period : ", end - cent)
        #     x = x2

        x, edge_index = data.x, data.edge_index
        x = self.quant(x)
        # x, temp_min, temp_offset = qunt_op(x, B)
        x = self.convs[0](x, edge_index)
        x = F.relu(x)
        x = self.dequant(x)
        # print(x)
        x = self.quant(x)
        # print(x)
        # x, temp_min, temp_offset = qunt_op(x, B)
        x = self.convs[1](x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # x = dequnt_op(x, temp_min, temp_offset, B)
        x = self.dequant(x)
        return F.log_softmax(x, dim=1)


device = 'cpu'
# device = 'cuda:0'
device = torch.device(device)
model = GraphSAGE()
# model.cuda()
model.to(device)
model.qconfig = torch.quantization.get_default_qconfig('x86')
learning_rate = 1e-2
weight_decay = 5e-3
epochs = 2
opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)



data = dataset[0]
data.to(device)
losses, val_accs = [], []
loss_fn = nn.NLLLoss()
record_acc = []

model_prepare = torch.quantization.prepare_qat(model, inplace=True)
model_prepare(data)
model_con = torch.quantization.convert(model_prepare)


def train():
    model.train()
    opt.zero_grad()
    out = model(data)
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    opt.step()

@torch.no_grad()
def test():
    model.eval()
    pred = model(data).max(dim=1)[1]
    correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / data.test_mask.sum()
    print('Epoch: {:03d}, Val Acc.: {:.3f}'.format(epoch, acc))

if __name__ == "__main__":
    for epoch in range(epochs):
        train()
        if epoch % 10 == 0 and epoch != 0:
            test()
    # print("Maximum accuracy: {0}".format(max(val_accs)))
    # print("Minimum loss: {0}".format(min(losses)))