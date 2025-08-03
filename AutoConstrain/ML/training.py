import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader, Batch
from tqdm import tqdm
import os

def collate_fn(batch):
    return Batch.from_data_list(batch)

def build_labels(sketch, candidates, id_to_idx, role_map, no_constraint_label=0):
    labels = torch.full((candidates.size(0),), no_constraint_label, dtype=torch.long)
    constraint_pairs = {}
    for constraint in sketch.constraints.values():
        entities = list(constraint.entities.values())
        c_type = constraint.type
        if len(entities) == 1:
            idx = id_to_idx.get(entities[0].id)
            if idx is not None:
                constraint_pairs[(idx, idx)] = role_map.get(c_type, no_constraint_label)
        elif len(entities) == 2:
            idx1 = id_to_idx.get(entities[0].id)
            idx2 = id_to_idx.get(entities[1].id)
            if idx1 is not None and idx2 is not None:
                pair = tuple(sorted([idx1, idx2]))
                constraint_pairs[pair] = role_map.get(c_type, no_constraint_label)
    for i, (n1, n2) in enumerate(candidates.tolist()):
        pair = tuple(sorted([n1, n2]))
        if pair in constraint_pairs:
            labels[i] = constraint_pairs[pair]
    return labels

def evaluate(model, dataloader, role_map, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            edge_logits, candidates, _ = model(data)
            if edge_logits is None or candidates is None:
                continue
            id_to_idx = {nid: idx for idx, nid in enumerate(data.x_id)}
            labels = build_labels(data.sketch, candidates, id_to_idx, role_map).to(device)
            loss = F.cross_entropy(edge_logits, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def train(model, train_dataset, val_dataset, optimizer, epochs=10, checkpoint_path="checkpoint.pt"):
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_correct = 0
        total_labels = 0

        for data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training"):
            data = data.to(model.device)
            optimizer.zero_grad()

            # Forward pass
            unary_logits, edge_logits = model(data)

            loss = 0

            # Handle edge logits
            if edge_logits is not None:
                sketch = data.sketch
                id_to_idx = {nid: idx for idx, nid in enumerate(data.x_id)}

                edge_index = data.edge_index
                edge_attrs = data.edge_attr
                labels = sketch.build_labels(edge_index, edge_attrs, id_to_idx).to(model.device)

                edge_loss = F.cross_entropy(edge_logits, labels)
                loss += edge_loss

                # Accuracy
                preds = edge_logits.argmax(dim=1)
                correct = (preds == labels).sum().item()
                total_correct += correct
                total_labels += len(labels)

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        accuracy = total_correct / total_labels if total_labels > 0 else 0
        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Accuracy = {accuracy:.4f}")

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} - Validation"):
                data = data.to(model.device)
                unary_logits, edge_logits = model(data)
                loss = 0

                if edge_logits is not None:
                    sketch = data.sketch
                    id_to_idx = {nid: idx for idx, nid in enumerate(data.x_id)}
                    edge_index = data.edge_index
                    edge_attrs = data.edge_attr
                    labels = sketch.build_labels(edge_index, edge_attrs, id_to_idx).to(model.device)
                    edge_loss = F.cross_entropy(edge_logits, labels)
                    loss += edge_loss

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}: Val Loss = {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_" + checkpoint_path)

        # Save current checkpoint
        torch.save(model.state_dict(), checkpoint_path)
