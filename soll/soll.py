import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
from typing import Tuple, Optional

class SOLL(nn.Module):
    """SOLL (Save Once, Load Later) encoding module.

    This implementation provides a lightweight, plug-and-play structural encoder
    that can be attached to a backbone model without rewriting the core pipeline.
    It exposes both spatial encoding and edge-path encoding in tensor form.
    """

    def __init__(self, max_path_distance: int = 5, edge_dim: Optional[int] = None):
        super().__init__()
        self.max_path_distance = max_path_distance
        self.edge_dim = edge_dim
        
        # Learnable spatial-distance coefficients.
        self.b = nn.Parameter(torch.randn(max_path_distance))
        
        # Learnable edge-path projection parameters.
        if edge_dim is not None:
            self.edge_vector = nn.Parameter(torch.randn(max_path_distance, edge_dim))
        
        # Lightweight in-memory cache: graph_hash -> (src_tensor, dst_tensor, path_tensor)
        self.register_buffer('cache', torch.empty(0))  # Marker buffer for module state.
        self.memory_dict = {}  

    def _get_graph_hash(self, edge_index: torch.Tensor, num_nodes: int) -> str:
        """Create a stable hash key for the current graph topology."""
        edges = tuple(sorted(zip(edge_index[0].cpu().tolist(), edge_index[1].cpu().tolist())))
        return str(edges) + f"_n{num_nodes}"

    def _compute_path_triplet(self, edge_index: torch.Tensor, num_nodes: int):
        """Compute and cache the path triplet, which is the core SOLL primitive."""
        graph_hash = self._get_graph_hash(edge_index, num_nodes)
        
        if graph_hash in self.memory_dict:
            return self.memory_dict[graph_hash]
        
        # Convert the graph to NetworkX and enumerate shortest paths.
        G = to_networkx(Data(edge_index=edge_index, num_nodes=num_nodes), to_undirected=True)
        
        src_list, dst_list, path_indices = [], [], []
        
        for src in range(num_nodes):
            try:
                node_paths = nx.single_source_shortest_path(G, src, cutoff=self.max_path_distance)
                for dst, path in node_paths.items():
                    if src == dst:
                        continue
                    edge_path = []
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        # Resolve the corresponding edge index for each hop.
                        edge_idx = torch.where((edge_index[0] == u) & (edge_index[1] == v))[0]
                        if len(edge_idx) > 0:
                            edge_path.append(edge_idx[0].item())
                    if edge_path:
                        src_list.append(src)
                        dst_list.append(dst)
                        path_indices.append(edge_path[:self.max_path_distance])
            except:
                continue
        
        # Materialize the collected path metadata as tensors.
        src_tensor = torch.tensor(src_list, dtype=torch.long)
        dst_tensor = torch.tensor(dst_list, dtype=torch.long)
        
        # Pad all paths to a fixed length for batched tensor operations.
        padded_paths = []
        for p in path_indices:
            if len(p) < self.max_path_distance:
                p = p + [-1] * (self.max_path_distance - len(p))
            padded_paths.append(p)
        path_tensor = torch.tensor(padded_paths, dtype=torch.long)
        
        triplet = (src_tensor, dst_tensor, path_tensor)
        self.memory_dict[graph_hash] = triplet
        return triplet

    def spatial_encoding(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Compute the tensorized spatial encoding."""
        device = edge_index.device
        # Build the sparse adjacency matrix.
        values = torch.ones(edge_index.shape[1], dtype=torch.float32, device=device)
        adj = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes), device=device)
        adj = adj + torch.eye(num_nodes, device=device).to_sparse()
        
        spatial_matrix = torch.zeros((num_nodes, num_nodes), device=device)
        adj_last = torch.zeros((num_nodes, num_nodes), device=device).to_sparse()
        
        for i in range(self.max_path_distance):
            spatial_matrix += (adj.to_dense() - adj_last.to_dense()) * self.b[i]
            adj_last = adj
            adj = torch.sparse.mm(adj, adj)
            adj = torch.sparse_coo_tensor(
                adj.indices(),
                torch.clamp(adj.values(), max=1.0),
                adj.size(),
                device=device
            )
        return spatial_matrix

    def edge_encoding(self, edge_index: torch.Tensor, edge_attr: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Compute the tensorized SOLL edge encoding."""
        if not hasattr(self, 'edge_vector'):
            raise ValueError("edge_dim must be set in __init__ to use edge_encoding")
        
        device = edge_index.device
        triplet = self._compute_path_triplet(edge_index, num_nodes)
        src_tensor, dst_tensor, path_tensor = triplet
        
        if len(src_tensor) == 0:
            return torch.zeros((num_nodes, num_nodes), device=device)
        
        src_tensor = src_tensor.to(device)
        dst_tensor = dst_tensor.to(device)
        path_tensor = path_tensor.to(device)
        
        mask = (path_tensor != -1).float()
        path_tensor = torch.clamp(path_tensor, min=0)
        
        # Gather edge features along each shortest path.
        edge_features = edge_attr[path_tensor] * mask.unsqueeze(-1)
        
        # Perform the path-wise interaction in a fully vectorized manner.
        dot_products = (edge_features * self.edge_vector.unsqueeze(0)).sum(dim=-1)
        dot_products = dot_products.mean(dim=1)
        
        cij = torch.zeros((num_nodes, num_nodes), device=device)
        cij[src_tensor, dst_tensor] = dot_products
        
        return cij

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unified forward interface returning (spatial_matrix, edge_matrix)."""
        num_nodes = x.shape[0]
        spatial = self.spatial_encoding(edge_index, num_nodes)
        
        if edge_attr is not None and hasattr(self, 'edge_vector'):
            edge_mat = self.edge_encoding(edge_index, edge_attr, num_nodes)
        else:
            edge_mat = torch.zeros((num_nodes, num_nodes), device=x.device)
        
        return spatial, edge_mat

