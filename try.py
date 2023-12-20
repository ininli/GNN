from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='./tmp/cora/', name='Cora')
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
print(dataset[0])

