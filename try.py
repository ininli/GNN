from torch_geometric.datasets import Planetoid
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import SAGEConv


dataset = Planetoid(root='./tmp/cora/', name='Cora')
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
'''
class GraphSAGE(nn.Module):
    def __init__(self):
        super(GraphSAGE, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(1433, 64))
        self.convs.append(SAGEConv(64, 32))
        self.convs.append(SAGEConv(32, 32))

        self.dropout = 0.5
        self.post_mp = nn.Sequential(
            nn.Linear(32, 32),
            nn.Dropout(self.dropout),
            nn.Linear(32, 7)
        )

    def forward(self, data):
        # x, edge_index, batch = data.x, data.edge_index, data.batch

        x, edge_index = data.x, data.edge_index

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = int(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.post_mp(x)

        return F.log_softmax(x, dim=1)

    

model = GraphSAGE()
filter_fn = filter(lambda p: p.requires_grad, model.parameters())
learning_rate = 1e-2
weight_decay = 5e-3
opt = torch.optim.Adam(filter_fn, lr=learning_rate, weight_decay=weight_decay)

data = dataset[0]
losses, val_accs = [], []
loss_fn = nn.NLLLoss()
for epoch in range(200):
    model.train()
    opt.zero_grad()
    out = model(data)
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    opt.step()
    losses.append(loss.item())

    if epoch % 10 == 0:
        model.eval()  # 设置模型为验证模式

        with torch.no_grad():
            # max(dim=1) returns (values, indices) tuple; only need indices
            pred = model(data).max(dim=1)[1]
            correct = float(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
            acc = correct / data.val_mask.sum().item()

        val_accs.append(acc)
        print('Epoch: {:03d}, Loss: {:.5f}, Val Acc.: {:.3f}'.format(epoch, loss.item(), acc))
        
    else:
        val_accs.append(val_accs[-1])  # 保证画图时验证集的尺度一直
        
print("Maximum accuracy: {0}".format(max(val_accs)))
print("Minimum loss: {0}".format(min(losses)))
