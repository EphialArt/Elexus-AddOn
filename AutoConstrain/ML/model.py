import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class ConstraintPredictorGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_edge_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, num_edge_classes)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        device = x.device

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Build set of existing edges (sorted tuples, undirected)
        existing_edges = set(tuple(sorted(e)) for e in edge_index.t().tolist())
        num_nodes = x.size(0)

        candidates = set(existing_edges)  # Start with all existing edges

        # Add all possible self-loops (unary constraints)
        for i in range(num_nodes):
            candidates.add((i, i))

        # Add all possible pairs (i, j) with i < j
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                candidates.add((i, j))

        candidates = torch.tensor(list(candidates), dtype=torch.long, device=device)

        node_feats = torch.cat([x[candidates[:, 0]], x[candidates[:, 1]]], dim=1)
        edge_logits = self.edge_mlp(node_feats)

        return edge_logits, candidates, x