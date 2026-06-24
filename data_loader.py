# data_loader.py
import torch
from torch_geometric.datasets import HGBDataset, DBLP,IMDB
from torch_geometric.transforms import ToUndirected
from torch_geometric.data import HeteroData
from torch_sparse import spspmm, coalesce, SparseTensor
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from typing import List, Tuple
import pandas as pd

def load_dataset(args, root='data'):
    if args.dataset == 'DBLP':
        dataset = DBLP(root=f'{root}/DBLP', transform=ToUndirected())
        data = dataset[0]
        target_ntype = 'author'
        metadata = [edge_type for edge_type in data.metadata()[1] if 'conference' not in edge_type]
    elif args.dataset == 'ACM':
        dataset = HGBDataset(root=root, name='ACM',transform=ToUndirected())
        data = dataset[0]
        target_ntype = 'paper'
        metadata = [edge_type for edge_type in data.metadata()[1] if 'term' not in edge_type]

    elif args.dataset == 'IMDB':
        dataset = IMDB(root='data/IMDB')
        data = dataset[0]
        target_ntype = 'movie'
        metadata = data.metadata()[1]

    data = generate_masks(data, target_ntype, args, seed=42)
    train_idx = data[target_ntype].train_mask
    val_idx = data[target_ntype].val_mask
    test_idx = data[target_ntype].test_mask
    y = data[target_ntype].y

    return data,metadata,y,train_idx,val_idx,test_idx,target_ntype


def generate_masks(data, target_ntype, args, seed=42):
    train_ratio = args.train_ratio
    num_nodes = data[target_ntype].num_nodes
    indices = np.arange(num_nodes)
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_size = int(train_ratio * num_nodes)
    val_size = (num_nodes - train_size) // 2
    test_size = num_nodes - train_size - val_size

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data[target_ntype].train_mask = train_mask
    data[target_ntype].val_mask = val_mask
    data[target_ntype].test_mask = test_mask

    return data

def adj_to_bias_torch_from_edge_index(edge_index: torch.Tensor, num_nodes: int, nhood=1, device='cpu'):
    adj = torch.zeros((num_nodes, num_nodes), device=device)
    adj[edge_index[0], edge_index[1]] = 1
    mt = torch.eye(num_nodes, device=device)
    adj_eye = adj + torch.eye(num_nodes, device=device)
    for _ in range(nhood):
        mt = torch.matmul(mt, adj_eye)
    mt = (mt > 0).float()
    bias = -1e9 * (1.0 - mt)
    return bias

def to_adj(edge_index, src_size, dst_size):
    row, col = edge_index
    values = np.ones(row.shape[0])
    return sp.coo_matrix((values, (row.numpy(), col.numpy())), shape=(src_size, dst_size))

def find_metapaths(metadata, center, max_depth=4):
    from collections import defaultdict
    edge_dict = defaultdict(list)
    reverse_edges = set()

    # 记录所有边和反向可达性
    for start, _, end in metadata:
        edge_dict[start].append([start, _, end])
        reverse_edges.add((end, start))  # 用于验证是否存在反向路径

    results = []

    def dfs(path):
        last_node = path[-1][2]
        if last_node == center and len(path) > 1:
            results.append(path.copy())
            return
        if len(path) >= max_depth:
            return
        for next_edge in edge_dict.get(last_node, []):
            prev_end = path[-1][2]
            next_start, _, next_end = next_edge
            if prev_end != next_start:
                continue
            # 必须有回环边（即这个跳跃能回到中心）
            if (next_end, next_start) not in reverse_edges:

                continue

            path.append(next_edge)
            dfs(path)
            path.pop()

    # 从中心节点出发，只保留能回到center的方向
    for edge in edge_dict[center]:
          # 只有能回来的边才开始搜索
        dfs([edge])

    return results


def extract_metapath_key(path):
    """将[['author', 'to', 'paper'], ...] 转换为字符串 'APTPA'"""
    key = [triplet[0][0].upper() for triplet in path]
    key.append(path[-1][2][0].upper())  # 添加最后一跳的目标节点首字母
    return ''.join(key)

def compute_metapath_bias_dict(paths, data, center_node_type):
    """
    paths: List of metapaths, each is a list of triplets
    Returns: dict of {metapath_key: (bias_tensor, edge_index_tensor)}
    """
    bias_dict = {}

    for path in paths:
        metapath_key = extract_metapath_key(path)

        # 构建邻接矩阵链
        adj_matrices = []
        for triplet in path:
            src, rel, dst = triplet
            try:
                edge_index = data[src, rel, dst].edge_index
            except KeyError:
                raise ValueError(f"Edge ({src}, {rel}, {dst}) not found in data.")

            row_N = data[src].num_nodes
            col_N = data[dst].num_nodes
            adj = to_adj(edge_index, row_N, col_N)
            adj_matrices.append(adj)

        # 链式矩阵乘法
        adj_final = adj_matrices[0]
        for i in range(1, len(adj_matrices)):
            adj_final = adj_final @ adj_matrices[i]

        # 去掉自连接
        adj_final.setdiag(0)
        adj_final.eliminate_zeros()
        adj_final = adj_final.tocoo()

        # 转为 PyG 格式
        edge_index = torch.tensor(np.vstack((adj_final.row, adj_final.col)), dtype=torch.long)
        bias = adj_to_bias_torch_from_edge_index(edge_index, num_nodes=data[center_node_type].num_nodes)

        # 存入字典：允许一个 key 有多个版本（用列表装起来）
        if metapath_key not in bias_dict:
            bias_dict[metapath_key] = []
        bias_dict[metapath_key].append(bias)

    return bias_dict, list(bias_dict.keys())


def build_han_data(args):
    # 加载数据
    data,metadata,y,train_idx,val_idx,test_idx,target_ntype = load_dataset(args)

    paths =find_metapaths(metadata,target_ntype, max_depth=4)
    # print(paths)
    bias_dict, keys = compute_metapath_bias_dict(
        paths=paths,
        data=data,
        center_node_type=target_ntype
    )
    print(keys)

    # # 模拟 HAN 的三分支：feature 重复三份
    feature = data[target_ntype].x
    features_list = [feature,feature]
    adj = [bias_dict[keys[0]][0],bias_dict[keys[1]][0]]

    return features_list, adj, y, train_idx, val_idx, test_idx
