from typing import Tuple
import torch
from torch import nn
from torch_geometric.utils import degree


class CentralityEncoding(nn.Module):
    def __init__(self, max_in_degree: int, max_out_degree: int, node_dim: int):
        """
        :param max_in_degree: max in degree of nodes
        :param max_out_degree: max out degree of nodes
        :param node_dim: hidden dimensions of node features
        """
        super().__init__()
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.node_dim = node_dim
        self.z_in = nn.Parameter(torch.randn((max_in_degree, node_dim)))
        #self.z_out = nn.Parameter(torch.randn((max_out_degree, node_dim)))

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param edge_index: edge_index of graph (adjacency list)
        :return: torch.Tensor, node embeddings after Centrality encoding
        """
        num_nodes = x.shape[0]

        in_degree = self.decrease_to_max_value(degree(index=edge_index[1], num_nodes=num_nodes).long(),
                                               self.max_in_degree - 1) # 每个节点的入度

        # 度中心性编码
        #x += self.z_in[in_degree] + self.z_out[out_degree] # 将每个节点度的数值作为索引，挑选z_in或z_out的每行，形成每个节点的嵌入
        x = x.to(self.z_in.device)
        in_degree = in_degree.to(self.z_in.device)
        x += self.z_in[in_degree]

        return x

    def decrease_to_max_value(self, x, max_value):
        "限制节点度的最大值"
        x[x > max_value] = max_value

        return x


class SpatialEncoding(nn.Module):
    def __init__(self, max_path_distance: int):
        """
        :param max_path_distance: max pairwise distance between nodes
        """
        super().__init__()
        self.max_path_distance = max_path_distance
        self.b = nn.Parameter(torch.randn(self.max_path_distance))

    def forward(self, x: torch.Tensor, paths) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param paths: pairwise node paths
        :return: torch.Tensor, spatial Encoding matrix
        """
        #spatial_matrix = torch.zeros((x.shape[0], x.shape[0])).to(next(self.parameters()).device) # (num_nodes, num_nodes)
        spatial_matrix = torch.zeros((x.shape[0], x.shape[0]))
        """
        paths[0]:{0: [0], 1: [0, 1], 10: [0, 10], 2: [0, 1, 2], 9: [0, 10, 9], 11: [0, 10, 11], 3: [0, 1, 2, 3], 
                8: [0, 10, 9, 8], 6: [0, 10, 11, 6], 4: [0, 1, 2, 3, 4], 7: [0, 10, 9, 8, 7], 5: [0, 10, 11, 6, 5]}
        paths[src = 0][dst = 10] = [0, 10]
        """
        for src in paths:
            for dst in paths[src]:
                spatial_matrix[src][dst] = self.b[min(len(paths[src][dst]), self.max_path_distance) - 1] # 索引从 0 到 max_path_distance-1

        return spatial_matrix


class EdgeEncoding(nn.Module):
    def __init__(self, edge_dim: int, max_path_distance: int):
        """
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.edge_dim = edge_dim
        self.max_path_distance = max_path_distance
        self.edge_vector = nn.Parameter(torch.randn(self.max_path_distance, self.edge_dim))

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_paths) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param edge_paths: pairwise node paths in edge indexes
        :return: torch.Tensor, Edge Encoding matrix
        """
        #cij = torch.zeros((x.shape[0], x.shape[0])).to(next(self.parameters()).device)
        cij = torch.zeros((x.shape[0], x.shape[0]))

        for src in edge_paths:
            for dst in edge_paths[src]:
                path_ij = edge_paths[src][dst][:self.max_path_distance] # 获取最短路径（截断）
                weight_inds = [i for i in range(len(path_ij))]
                cij[src][dst] = self.dot_product(self.edge_vector[weight_inds], edge_attr[path_ij]).mean()

        cij = torch.nan_to_num(cij) # 路径可能无数值，后续计算产生NaN

        return cij

    def dot_product(self, x1, x2) -> torch.Tensor:
        return (x1 * x2).sum(dim=1)  # 沿着第二维度求和，即对二维张量的每行求和（返回值为一维张量，一行多列）


class EdgeEncodingPlus(nn.Module):
    def __init__(self, edge_dim: int, max_path_distance: int):
        """
        :param edge_dim: edge feature matrix number of dimension
        :param max_path_distance: maximum path distance for encoding
        """
        super().__init__()
        self.edge_dim = edge_dim
        self.max_path_distance = max_path_distance
        self.edge_vector = nn.Parameter(
            torch.randn(self.max_path_distance, self.edge_dim))  # Shape: (max_path_distance, edge_dim)

    def forward(self, num_nodes:int, edge_attr: torch.Tensor, edge_paths: list) -> torch.Tensor:
        """
        :param x: node feature matrix, shape (num_nodes, node_dim)
        :param edge_attr: edge feature matrix, shape (num_edges, edge_dim)
        :param edge_paths: dictionary of edge paths between nodes
        :return: Edge Encoding matrix (num_nodes, num_nodes)
        """

        device = self.edge_vector.device  # Move to correct device
        src_tensor, dst_tensor, path_tensor = edge_paths[0], edge_paths[1], edge_paths[2]
        src_tensor = src_tensor.to(device)
        dst_tensor = dst_tensor.to(device)
        path_tensor = path_tensor.to(device)

        # If no paths exist, return zero matrix
        if len(path_tensor) == 0:
            return torch.zeros((num_nodes, num_nodes), device=device)

        # Prepare tensors for batch processing
        #src_tensor = torch.tensor(src_list, device=device)
        #dst_tensor = torch.tensor(dst_list, device=device)
        #path_tensor = torch.tensor(path_indices, device=device)

        mask = (path_tensor != -1).float()
        path_tensor = torch.clamp(path_tensor, min=0)  # turn -1 to 0

        # Gather edge attributes for each path step
        edge_features = edge_attr[path_tensor]  # Shape: (num_paths, truncated_path_len, edge_dim)
        edge_features = edge_features * mask.unsqueeze(-1)

        # Compute dot products along path steps and average
        dot_products = (edge_features * self.edge_vector.unsqueeze(0)).sum(dim=-1)  # Shape: (num_paths, truncated_path_len)
        dot_products = dot_products.mean(dim=1)

        # Construct the (num_nodes, num_nodes) edge encoding matrix
        cij = torch.zeros((num_nodes, num_nodes), device=device)
        cij[src_tensor, dst_tensor] = dot_products  # Populate with computed values

        return cij


class GraphormerAttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int, edge_dim: int, max_path_distance: int):
        """
        :param dim_in: node feature matrix input number of dimension
        :param dim_q: query node feature matrix input number dimension
        :param dim_k: key node feature matrix input number of dimension
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()

        self.edge_encoding = EdgeEncodingPlus(edge_dim, max_path_distance)

        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

        self.attn_raw = nn.Parameter(torch.tensor([1.0, 1.0]))

    def forward(self,
                x: torch.Tensor,
                edge_attr: torch.Tensor,
                b: torch.Tensor,
                edge_paths,
                ptr=None) -> torch.Tensor:
        """
        :param query: node feature matrix
        :param key: node feature matrix
        :param value: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after attention operation
        """
    
        batch_mask_neg_inf = torch.full(size=(x.shape[0], x.shape[0]), fill_value=-1e6,device=x.device)
        batch_mask_zeros = torch.zeros(size=(x.shape[0], x.shape[0]), device=x.device)

        if type(ptr) == type(None):
            batch_mask_neg_inf = torch.ones(size=(x.shape[0], x.shape[0])).to(next(self.parameters()).device)
            batch_mask_zeros += 1
        else:
            # 批图的mask,邻接矩阵以对角阵组合
            for i in range(len(ptr) - 1):
                batch_mask_neg_inf[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = 1
                batch_mask_zeros[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = 1

        query = self.q(x)
        key = self.k(x)
        value = self.v(x)

        c = self.edge_encoding(x.shape[0], edge_attr, edge_paths)
        a = self.compute_a(key, query, ptr)
        """
        mask coding
        考虑到pyg的一个batch里会有多个原子的图，不同的图之间不存在注意力值，需要mask coding机制
        a: key*qurry   b: 空间编码   c：边编码 
        """
        weights = torch.softmax(self.attn_raw, dim=0)
        b_weight, c_weight = weights[0], weights[1]

        a = (a + b_weight * b.to(a.device) + c_weight * c.to(a.device)) * batch_mask_neg_inf
        #a = (a + b.to(a.device) + c.to(a.device)) * batch_mask_neg_inf
        softmax = torch.softmax(a, dim=-1) * batch_mask_zeros # e^(-inf) ——> 0
        x = softmax.mm(value)

        return x

    def compute_a(self, key, query, ptr=None):
        "Query-Key product(normalization)"
        if type(ptr) == type(None):
            a = query.mm(key.transpose(0, 1)) / query.size(-1) ** 0.5
        else:
            a = torch.zeros((query.shape[0], query.shape[0]), device=key.device)
            for i in range(len(ptr) - 1):
                a[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = query[ptr[i]:ptr[i + 1]].mm(
                    key[ptr[i]:ptr[i + 1]].transpose(0, 1)) / query.size(-1) ** 0.5

        return a


class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int, edge_dim: int, max_path_distance: int):
        """
        :param num_heads: number of attention heads
        :param dim_in: node feature matrix input number of dimension
        :param dim_q: query node feature matrix input number dimension
        :param dim_k: key node feature matrix input number of dimension
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [GraphormerAttentionHead(dim_in, dim_q, dim_k, edge_dim, max_path_distance) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self,
                x: torch.Tensor,
                edge_attr: torch.Tensor,
                b: torch.Tensor,
                edge_paths,
                ptr) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after all attention heads
        """
        return self.linear(
            torch.cat([
                attention_head(x, edge_attr, b, edge_paths, ptr) for attention_head in self.heads
            ], dim=-1)
        )


class GraphormerEncoderLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, num_heads, max_path_distance, dropout_rate = 0.3):
        """
        :param node_dim: node feature matrix input number of dimension
        :param edge_dim: edge feature matrix input number of dimension
        :param num_heads: number of attention heads
        """
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads

        self.attention = GraphormerMultiHeadAttention(
            dim_in=node_dim,
            dim_k=node_dim,
            dim_q=node_dim,
            num_heads=num_heads,
            edge_dim=edge_dim,
            max_path_distance=max_path_distance,
        )
        self.ln_1 = nn.LayerNorm(node_dim)
        self.ln_2 = nn.LayerNorm(node_dim)
        self.ff = nn.Linear(node_dim, node_dim)
        #Transformer-style FFN: Linear -> GELU -> Dropout -> Linear -> Dropout
        # self.ff = nn.Sequential(
        #     nn.Linear(node_dim, node_dim),
        #     nn.GELU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(node_dim, node_dim),
        #     nn.Dropout(0.1)
        # )
        self.dropout = nn.Dropout(dropout_rate)  # 增加 Dropout

    def forward(self,
                x: torch.Tensor,
                edge_attr: torch.Tensor,
                b: torch,
                edge_paths,
                ptr) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h′(l) = MHA(LN(h(l−1))) + h(l−1)
        h(l) = FFN(LN(h′(l))) + h′(l)

        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after Graphormer layer operations

        # version without dropout
        x_prime = self.attention(self.ln_1(x), edge_attr, b, edge_paths, ptr) + x
        x_new = self.ff(self.ln_2(x_prime)) + x_prime
        """

        # Multi-Head Attention with Dropout
        x_prime = self.attention(self.ln_1(x), edge_attr, b, edge_paths, ptr)
        x_prime = x_prime + x  # 残差连接

        # Feed-Forward Network with Dropout
        x_new = self.ff(self.ln_2(x_prime))
        x_new = self.dropout(x_new)
        x_new = x_new + x_prime  # 残差连接

        return x_new
