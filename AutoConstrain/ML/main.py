from dataloader import load_dataset, visualize_graph
from model import ConstraintPredictorGNN
import torch
from training import train
from creategraph import role_map
from training import epoch_no
from mldataset import Dataset

train_set = Dataset("AutoConstrain/Dataset/train", drop_rate=1.0)
val_set = Dataset("AutoConstrain/Dataset/val", drop_rate=1.0, seed=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConstraintPredictorGNN(in_channels=3, hidden_channels=64, num_edge_classes=len(role_map)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


train(model, train_set, val_set, role_map, optimizer, device=device, epochs=500)