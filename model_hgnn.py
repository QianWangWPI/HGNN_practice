# model_hgnn.py
import torch
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv, Linear, GATConv
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from attention_layer import HANLayer


class GraphSAGE(torch.nn.Module):
    def __init__(self, metadata, args):
        super(GraphSAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(2):
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), args.hidden_channels)
                for edge_type in metadata}, aggr='sum')
            self.convs.append(conv)
        self.lin = Linear(args.hidden_channels, args.out_channels)
        self.target_ntype = args.target_ntype

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        out = self.lin(x_dict[self.target_ntype])
        return out


class HAN(nn.Module):
    def __init__(self, num_metapaths, in_dim, hidden_dim, out_dim, num_heads=8, dropout=0.5):
        super(HAN, self).__init__()
        self.han_layer = HANLayer(num_metapaths, in_dim, hidden_dim, num_heads, dropout)
        self.predict = nn.Linear(hidden_dim*num_heads, out_dim)

    def forward(self, inputs_list, bias_mat_list):
        h, att = self.han_layer(inputs_list, bias_mat_list)
        out = self.predict(h).squeeze(0)  # [B, N, out_dim]
        return out, att

