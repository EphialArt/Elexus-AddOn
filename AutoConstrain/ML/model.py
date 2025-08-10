import torch
import torch.nn.functional as F
from torch_geometric.nn import NNConv

class ConstraintPredictorGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_attr_dim, num_edge_classes):
        super().__init__()
        self.conv1 = NNConv(
            in_channels,
            hidden_channels,
            torch.nn.Sequential(
                torch.nn.Linear(edge_attr_dim, hidden_channels * in_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels * in_channels, hidden_channels * in_channels)
            )
        )
        self.conv2 = NNConv(
            hidden_channels,
            hidden_channels,
            torch.nn.Sequential(
                torch.nn.Linear(edge_attr_dim, hidden_channels * hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels * hidden_channels, hidden_channels * hidden_channels)
            )
        )
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, num_edge_classes)
        )

    def forward(self, data, override_candidates=None):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        device = x.device

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))

        num_nodes = x.size(0)

        if override_candidates is not None:
            candidates = override_candidates
        else:
            candidates = []
            for i in range(num_nodes):
                for j in range(i, num_nodes):
                    candidates.append((i, j))
            candidates = torch.tensor(candidates, dtype=torch.long, device=device)

        node_feats = torch.cat([x[candidates[:, 0]], x[candidates[:, 1]]], dim=1)
        edge_logits = self.edge_mlp(node_feats)

        return edge_logits, candidates, x

