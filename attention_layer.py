import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.6, alpha=0.2, residual=False):
        super(GraphAttentionLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.f1 = nn.Linear(out_dim, 1)
        self.f2 = nn.Linear(out_dim, 1)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = dropout
        self.residual = residual

    def forward(self, x, bias_mat):
        # x: [B, N, F], bias_mat: [B, N, N]
        h = self.fc(x)  # Linear transform -> [B, N, out_dim]
        f1 = self.f1(h)  # [B, N, 1]
        f2 = self.f2(h)  # [B, N, 1]
        logits = f1 + f2.T # broadcasting -> [B, N, N]

        coefs = F.softmax(self.leakyrelu(logits + bias_mat), dim=-1)  # masked softmax attention
        coefs = F.dropout(coefs, self.dropout, training=self.training)
        h = F.dropout(h, self.dropout, training=self.training)
        vals = torch.matmul(coefs, h)  # attention-weighted sum -> [B, N, out_dim]

        if self.residual:
            if x.shape[-1] != vals.shape[-1]:
                vals += self.fc(x)  # project to match shape
            else:
                vals += x
        return F.elu(vals)

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        # z: [B, M, D], M = # metapaths
        w = self.project(z)  # [B, M, 1]
        beta = torch.softmax(w, dim=1)  # 在 metapath 维度 softmax
        return (beta * z).sum(1), beta  # 输出: [N, D], [N, M, 1]

class HANLayer(nn.Module):
    def __init__(self, num_metapaths, in_dim, out_dim, num_heads=8, dropout=0.6):
        super(HANLayer, self).__init__()
        self.attn_heads = nn.ModuleList([
            nn.ModuleList([
                GraphAttentionLayer(in_dim, out_dim, dropout=dropout)
                for _ in range(num_heads)
            ])
            for _ in range(num_metapaths)
        ])
        self.semantic_attention = SemanticAttention(out_dim*num_heads, out_dim // 2)

    def forward(self, inputs_list, bias_mat_list):
        # inputs_list: list of [N, F], bias_mat_list: list of [N, N]
        metapath_outs = []

        for i in range(len(inputs_list)):
            head_outs = [
                self.attn_heads[i][j](inputs_list[i], bias_mat_list[i])  # 每个head的输出 [N, D]
                for j in range(len(self.attn_heads[i]))
            ]
            h = torch.cat(head_outs, dim=-1)  # [N, num_heads * out_dim]
            metapath_outs.append(h.unsqueeze(1))

        metapath_outs = torch.cat(metapath_outs, dim=1)  # [M, N, D]
        # metapath_outs = metapath_outs.permute(1, 0, 2)  # [N, M, D] → 每个节点有 M 条语义路径表示

        out, att = self.semantic_attention(metapath_outs)  # out: [N, D], att: [N, M, 1]

        return out, att