import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import DBLP
from torch_geometric.nn import HANConv
from torch_geometric.transforms import ToUndirected

# 1. Load DBLP dataset
dataset = DBLP(root='./data', transform=ToUndirected())
data = dataset[0]
print(data)
# 2. Move data to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
for node_type in data.x_dict:
    data[node_type].x = data[node_type].x.to(device)
for edge_type in data.edge_index_dict:
    data[edge_type].edge_index = data[edge_type].edge_index.to(device)

y = data['author'].y.to(device)
train_mask = data['author'].train_mask.to(device)
val_mask = data['author'].val_mask.to(device)
test_mask = data['author'].test_mask.to(device)

# 3. Define a minimal HAN model
class HAN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, metadata):
        super().__init__()
        self.han_conv = HANConv(
            in_channels, hidden_channels,
            metadata, heads=8)
        self.lin = Linear(hidden_channels * 8, out_channels)  # heads=8

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.han_conv(x_dict, edge_index_dict)
        return self.lin(x_dict['author'])

model = HAN(
    in_channels=334,  # author特征是334维
    hidden_channels=64,
    out_channels=4,   # 4分类
    metadata=data.metadata()
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# 4. Training + Validation
for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    loss = F.cross_entropy(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = out.argmax(dim=1)
        val_acc = (pred[val_mask] == y[val_mask]).sum() / val_mask.sum()
        if epoch % 20 == 0 or epoch == 1:
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}')

# 5. Final Test
with torch.no_grad():
    pred = out.argmax(dim=1)
    test_acc = (pred[test_mask] == y[test_mask]).sum() / test_mask.sum()
    print(f'Test Accuracy: {test_acc:.4f}')
