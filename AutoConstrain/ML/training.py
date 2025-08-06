import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import os
from creategraph import visualize_graph, build_pyg_graph

CHECKPOINT_DIR = "AutoConstrain/ML/checkpoints"

def collate_fn(batch):
    partials, fulls = zip(*batch)
    batch_partial = Batch.from_data_list(partials)
    batch_full = Batch.from_data_list(fulls)

    batch_partial.x_id = sum([data.x_id for data in partials], [])
    batch_full.x_id = sum([data.x_id for data in fulls], [])
    
    return batch_partial, batch_full

def get_constraint_training_pairs(full_sketch, partial_sketch, id_to_idx, role_map, no_constraint_label=0, structural_labels={0,1,2}):
    constraint_pairs = {}
    constraint_set = set()
    structural_pairs = set()

    # Add positive (dropped) constraints excluding structural links
    for cid, constraint in full_sketch.constraints.items():
        if cid in partial_sketch.constraints:
            continue

        entities = list(constraint.entities.values())
        c_type = constraint.type
        label = role_map.get(c_type, no_constraint_label)
        if label in structural_labels:
            # Skip structural links
            continue

        if len(entities) == 1:
            idx = id_to_idx.get(entities[0].id)
            if idx is not None:
                constraint_pairs[(idx, idx)] = label
                constraint_set.add((idx, idx))
        elif len(entities) == 2:
            idx1 = id_to_idx.get(entities[0].id)
            idx2 = id_to_idx.get(entities[1].id)
            if idx1 is not None and idx2 is not None:
                pair = tuple(sorted((idx1, idx2)))
                constraint_pairs[pair] = label
                constraint_set.add(pair)
        else:
            continue

    # Build set of structural pairs to exclude them from negatives
    for cid, constraint in full_sketch.constraints.items():
        entities = list(constraint.entities.values())
        c_type = constraint.type
        label = role_map.get(c_type, no_constraint_label)
        if label not in structural_labels:
            continue

        if len(entities) == 1:
            idx = id_to_idx.get(entities[0].id)
            if idx is not None:
                structural_pairs.add((idx, idx))
        elif len(entities) == 2:
            idx1 = id_to_idx.get(entities[0].id)
            idx2 = id_to_idx.get(entities[1].id)
            if idx1 is not None and idx2 is not None:
                pair = tuple(sorted((idx1, idx2)))
                structural_pairs.add(pair)

    # Sample some negatives, skipping positives and structural pairs
    all_idxs = list(id_to_idx.values())
    negative_pairs = []
    max_negatives = len(constraint_pairs) * 2

    for i in range(len(all_idxs)):
        for j in range(i, len(all_idxs)):
            pair = (all_idxs[i], all_idxs[j])
            if pair in constraint_set:
                continue
            if pair in structural_pairs:
                # Skip structural links from negative samples
                continue

            negative_pairs.append(pair)
            if len(negative_pairs) >= max_negatives:
                break
        if len(negative_pairs) >= max_negatives:
            break

    # Combine
    all_pairs = list(constraint_pairs.keys()) + negative_pairs
    labels = [constraint_pairs.get(p, no_constraint_label) for p in all_pairs]

    candidates = torch.tensor(all_pairs, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    return candidates, labels

def train(model, train_dataset, val_dataset, role_map, optimizer, device, epochs=10):
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    best_val_loss = float("inf")
    num_edge_classes = len(role_map)
    
    # Optional: class weights to address imbalance
    class_weights = torch.ones(num_edge_classes, device=device)
    class_weights[0] = 0.2  # Lower weight for "no constraint" class (label 0)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_correct = 0
        total_labels = 0

        for partial_data, full_data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training"):
            if partial_data.edge_index.numel() == 0:
                continue
            partial_data = partial_data.to(device)
            full_data = full_data.to(device)

            optimizer.zero_grad()

            x_id = partial_data.x_id[0] if isinstance(partial_data.x_id[0], list) else partial_data.x_id
            id_to_idx = {nid: idx for idx, nid in enumerate(x_id)}

            candidates, labels = get_constraint_training_pairs(full_data.sketch[0], partial_data.sketch[0], id_to_idx, role_map)
            candidates = candidates.to(device)
            labels = labels.to(device)
            edge_logits, _, _ = model(partial_data, override_candidates=candidates)

            if edge_logits is None or candidates is None:
                print(f"Skipping data with None edge_logits or candidates in epoch {epoch + 1}")
                continue

            sketch = full_data.sketch[0] if isinstance(full_data.sketch, list) else full_data.sketch
            # print(id_to_idx)

            loss = F.cross_entropy(edge_logits, labels, weight=class_weights)

            if loss.item() > 20:
                # visualize_graph(partial_data)
                # print(f"High loss detected: {loss.item()}. Visualizing graph.")
                print(f"Skipping sketch due to high loss: {sketch.id}")
                continue

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            preds = edge_logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_labels += len(labels)

        avg_train_loss = total_train_loss / len(train_loader)
        accuracy = total_correct / total_labels if total_labels > 0 else 0
        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Accuracy = {accuracy:.4f}")

        # Validation
        model.eval()
        total_val_loss = 0
        total_good_batches = 0
        with torch.no_grad():
            for partial_data, full_data in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} - Validation"):
                if partial_data.edge_index.numel() == 0:
                    continue

                partial_data = partial_data.to(device)
                full_data = full_data.to(device)

                x_id = partial_data.x_id[0] if isinstance(partial_data.x_id[0], list) else partial_data.x_id
                id_to_idx = {nid: idx for idx, nid in enumerate(x_id)}

                candidates, labels = get_constraint_training_pairs(full_data.sketch[0], partial_data.sketch[0], id_to_idx, role_map)
                candidates = candidates.to(device)
                labels = labels.to(device)
                edge_logits, _, _ = model(partial_data, override_candidates=candidates)

                if edge_logits is None or candidates is None:
                    print(f"Skipping data with None edge_logits or candidates in epoch {epoch + 1}")
                    continue

                sketch = full_data.sketch[0] if isinstance(full_data.sketch, list) else full_data.sketch

                val_loss = F.cross_entropy(edge_logits, labels, weight=class_weights)
                if val_loss.item() > 20:
                    # visualize_graph(partial_data)
                    # print(f"High loss detected: {loss.item()}. Visualizing graph.")
                    # print(f"Skipping sketch due to high loss: {partial_data.sketch.id}")
                    continue
                total_val_loss += val_loss.item()
                total_good_batches += 1

        avg_val_loss = total_val_loss / total_good_batches if total_good_batches > 0 else float("inf")
        print(f"Epoch {epoch + 1}: Val Loss = {avg_val_loss:.4f}")
        with open("AutoConstrain/loss.txt", "a") as f:
            f.write(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}\n")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_model_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)

        # Save current checkpoint
        torch.save(model.state_dict(), checkpoint_path)

