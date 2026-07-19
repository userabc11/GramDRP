import torch
import torch.nn.functional as F
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from soll import SOLL     # important

class GCNWithSOLL(torch.nn.Module):
    def __init__(self, node_dim=9, hidden_dim=64, max_path_distance=5):
        super().__init__()
        # SOLL module
        self.soll = SOLL(max_path_distance=max_path_distance, edge_dim=3)
        
        self.conv = GCNConv(node_dim, hidden_dim)   
        self.lin = torch.nn.Linear(hidden_dim, 1)  

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        spatial_bias, edge_bias = self.soll(x, edge_index, edge_attr)

        # use attenton metrix form SOLL as node and edge weights
        # you can do even more complex operations here, like concatenation, gating, etc.
        # here is just a simple demo
        
        row, col = edge_index
        edge_weight = 1.0 + edge_bias[row, col].detach()

        x = x + x * spatial_bias.mean(dim=1, keepdim=True)

        x = self.conv(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        
        return x


def train():
    dataset = MoleculeNet(root='/tmp/MoleculeNet', name='ESOL')
    print(f"Load ESOL dataset, len: {len(dataset)} ")
    
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    model = GCNWithSOLL(node_dim=dataset.num_node_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print(" SOLL usage demo\n")
    
    for epoch in range(10):
        model.train()
        total_loss = 0
        
        for batch in loader:
            optimizer.zero_grad() 
            pred = model(batch)
            loss = F.mse_loss(pred.squeeze(), batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1:2d} | Loss: {total_loss/len(loader):.4f}")


if __name__ == "__main__":
    torch.manual_seed(42)
    train()