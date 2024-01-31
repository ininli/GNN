from torch_geometric.datasets import Planetoid
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn.conv import SAGEConv
from torch_geometric.nn.conv import SAGEConv
from torch_geometric.profile import *
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import memory_profiler
writer = SummaryWriter()

dataset = Planetoid(root='../../dataset/', name='Cora')

class GraphSAGE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(dataset.num_node_features, 16)
        self.conv2 = SAGEConv(16, dataset.num_classes)
    def forward(self, data):
    # def forward(self, x, edge_index):
    #     x = self.conv1(x, edge_index)
    #     x = F.relu(x)
    #     x = F.dropout(x, training=self.training)
    #     x = self.conv2(x, edge_index)

        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
device = torch.device(device)

model = GraphSAGE()

model.cpu()
print(get_model_size(model))
#print(summary(model, (1, 1433, 3)))
data = dataset[0].to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# size : 0.176MB
# all_parameters = 0
# for parm in model.parameters():
#     print(parm.nelement(), parm.element_size())
#     all_parameters += parm.nelement() * parm.element_size()
# all_buffers = 0
# for buffer in model.buffers():
#     print(buffer)
#     all_buffers += buffer.nelement() * buffer.element_size()

# all_parameters = all_parameters/1024**2
# all_buffers = all_buffers/1024**2

# print('parameter size: {:.3f}MB \nbuffer size : {:.3f}MB'.format(all_parameters, all_buffers))

#----------------------------------

# prof = profile(
#         activities=[ProfilerActivity.CPU],
#         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
#         on_trace_ready=tensorboard_trace_handler('./log/sage'),
#         record_shapes=True,
#         profile_memory=True
# )
# prof.start()

@memory_profiler.profile
def train():
    for epoch in range(3):
        model.train()
        opt.zero_grad()

        with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
            with record_function("model"):
                # out = model(data.x, data.edge_index)
                out = model(data)
            
                loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                opt.step()

        print(prof.key_averages().table(row_limit=5))
        # evaluate
        model.eval()
        with torch.no_grad():
            # pred = model(data.x, data.edge_index).max(dim=1)[1]
            pred = model(data).max(dim=1)[1]
            correct = float(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
            acc = correct / data.val_mask.sum().item()

        print('Epoch: {:03d}, Loss: {:.5f}, Val Acc.: {:.3f}'.format(epoch+1, loss.item(), acc))
writer.close()

train()
# @profileit()
# def train(model, x, edge_index, y):
#     model.train()
#     opt.zero_grad()
#     out = model(x, edge_index)
#     loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
#     loss.backward()
#     opt.step()
#     return float(loss)
# loss, stats = train(model, data.x, data.edge_index, data.y)
# print(stats)
