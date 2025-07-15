import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from MLDataset import MeshCADTokenDataset, BucketSampler
from Encode_Decode import MeshToCADModel
from utils import collate_fn, MAX_TOTAL_TOKENS
from tqdm import tqdm
import os
import torch.nn.functional as F
from torch import amp

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
BATCH_SIZE = 128
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EOS_TOKEN = 501
VOCAB_SIZE = 502
SAVE_EVERY_N_BATCHES = 10
TARGET_EFFECTIVE_BATCH_SIZE = 32
CHECKPOINT_DIR = "MeshtoCAD/ML/checkpoints"
RESUME_PATH =  None
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

train_dataset = MeshCADTokenDataset("MeshtoCAD/TokenizedJSON/train")
val_dataset = MeshCADTokenDataset("MeshtoCAD/TokenizedJSON/val")

token_lengths = [len(data.tokens) for data in train_dataset if data is not None]

bucket_sampler = BucketSampler(lengths=token_lengths, batch_size=BATCH_SIZE, max_tokens=MAX_TOTAL_TOKENS, shuffle=True)
train_loader = DataLoader(train_dataset, batch_sampler=bucket_sampler, collate_fn=collate_fn, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)

torch.cuda.empty_cache()

model = MeshToCADModel(
    mesh_in_channels=3,
    mesh_hidden=128,
    mesh_latent=256,
    vocab_size=VOCAB_SIZE,
    dec_hidden=256,
    dec_layers=2
).to(DEVICE)

token_pad_id = 500 
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=token_pad_id) 
 
value_loss_weight = 3.0
float_loss_weight = 1.0
int_loss_weight = 2.0
uuid_loss_weight = 1.0

mean_val = None
std_val = None

try:
    with open("mean.txt", "r") as f:
        mean_val = str(f.read().strip())
    with open("std.txt", "r") as f:
        std_val = str(f.read().strip())
except:
    pass

print(f"Loaded mean: {mean_val}, std: {std_val}")

if mean_val is None and std_val is None:
    all_values = []

    for batch in train_loader:
        if batch is None:
            continue

        tokens = batch.tokens_padded       # shape: (B, T)
        floats = batch.floats_padded       # shape: (B, T)

        float_mask = (tokens == 310) & (floats != -1e6)
        float_values = floats[float_mask]

        all_values.append(float_values)

    if len(all_values) == 0:
        raise RuntimeError("No float values found in training data.")

    all_values = torch.cat(all_values)

    valid_values = all_values[(all_values > -3e4) & (all_values < 3e4)]

    mean_val = float(valid_values.mean().item())
    std_val = float(valid_values.std().item())

    print(f"mean: {mean_val}, std: {std_val}")
    print(f"max: {valid_values.max().item()}, min: {valid_values.min().item()}")


with open("mean.txt", "w") as f:
    f.write(str(mean_val))

with open("std.txt", "w") as f:
    f.write(str(std_val))


print(f"Value mean: {mean_val}, std: {std_val}")

def normalize(tensor, mean, std):
    tensor = torch.clamp(tensor, min=-1e5, max=1e5)
    return (tensor - mean) / std

def denormalize(tensor, mean, std):
    return tensor * std + mean

def evaluate(model, loader, criterion, device, vocab_size, mean_val, std_val):
    model.eval()
    total_loss = 0
    total_value_loss = 0
    total_token_loss = 0
    total_float_loss = 0
    total_int_loss = 0
    total_uuid_loss = 0
    mean_val, std_val = float(mean_val), float(std_val)
    valid_batches = 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            valid_batches += 1
            batch = batch.to(device)
            tokens = batch.tokens_padded
            floats = batch.floats_padded
            ints = batch.ints_padded
            uuids = batch.uuids_padded

            logits, float_preds, int_preds, uuid_preds = model(tokens[:, :-1], floats[:, :-1], ints[:, :-1], uuids[:, :-1], batch)

            token_targets = tokens[:, 1:]
            float_targets = floats[:, 1:]
            int_targets = ints[:, 1:]
            uuid_targets = uuids[:, 1:]
            token_loss = criterion(logits.reshape(-1, vocab_size), token_targets.reshape(-1))

            if token_loss.item() > 200:
                logits_flat = logits.reshape(-1, VOCAB_SIZE)
                token_targets_flat = token_targets.reshape(-1)
                token_loss_per_token = F.cross_entropy(logits_flat, token_targets_flat, ignore_index=token_pad_id, reduction='none')
                token_loss_per_sample = token_loss_per_token.view(tokens.shape[0], -1).mean(dim=1)
                unstable_indices = (token_loss_per_sample > 200).nonzero(as_tuple=True)[0].tolist()

                with open("Broken.txt", "a") as f:
                    files = [d.mesh_file for i, d in enumerate(batch.to_data_list()) if i in unstable_indices]
                    f.write(f"Files: {files}\n")
                continue
                        
            float_mask = (token_targets == 310) & (float_targets != -1e6)
            int_mask = (token_targets == 311) & (int_targets != -1)
            uuid_mask = (token_targets == 312) & (uuid_targets != -1)

            float_loss = torch.tensor(0.0, device=device)
            int_loss = torch.tensor(0.0, device=device)
            uuid_loss = torch.tensor(0.0, device=device)

            # Normalize for loss calculation
            if float_mask.any():
                norm_preds = normalize(float_preds[float_mask], mean_val, std_val)
                norm_targets = normalize(float_targets[float_mask], mean_val, std_val)
                float_loss = F.l1_loss(norm_preds, norm_targets)

            if int_mask.any():
                int_loss = F.cross_entropy(int_preds[int_mask], int_targets[int_mask])

            if uuid_mask.any():
                uuid_loss = F.cross_entropy(uuid_preds[uuid_mask], uuid_targets[uuid_mask])

            value_loss = float_loss_weight * float_loss + int_loss * int_loss_weight + uuid_loss * uuid_loss_weight
            loss = token_loss + value_loss_weight * value_loss

            total_loss += loss.item()
            total_token_loss += token_loss.item()
            total_value_loss += value_loss.item()
            total_float_loss += float_loss.item()
            total_int_loss += int_loss.item()
            total_uuid_loss += uuid_loss.item()

    if valid_batches == 0:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
    return total_loss / valid_batches, total_token_loss / valid_batches, total_value_loss / valid_batches, total_float_loss / valid_batches, total_int_loss / valid_batches, total_uuid_loss / valid_batches



best_val_loss = float('inf')
batch = next(iter(train_loader))
print("batch has tokens_padded?", hasattr(batch, "tokens_padded"))

start_epoch = 0
if RESUME_PATH is not None and os.path.exists(RESUME_PATH):
    checkpoint = torch.load(RESUME_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint.get('epoch', 0)
    print(f"Resumed from checkpoint: {RESUME_PATH}, starting at epoch {start_epoch}")

    val_loss, val_token_loss, val_value_loss, val_float_loss, val_int_loss, val_uuid_loss = evaluate(model, val_loader, criterion, DEVICE, VOCAB_SIZE, mean_val, std_val)
    print(f"Validation Token Loss: {val_token_loss:.4f}, Validation Value Loss: {val_value_loss:.4f}, Validation Float Loss: {val_float_loss:.4f}, Validation Int Loss: {val_int_loss:.4f}, Validation UUID Loss: {val_uuid_loss:.4f}")
else:
    print("Starting training from scratch.")

for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_loss = 0
    total_token_loss = 0
    total_value_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    valid_batches = 0
    mean_val, std_val = float(mean_val), float(std_val)
    num_batches = len(progress_bar)
    save_start = ((num_batches - 40) // 100) * 100
    
    for batch_idx, batch in enumerate(progress_bar):
        if batch is None:
            continue
        valid_batches += 1
        batch = batch.to(DEVICE)
        
        tokens = batch.tokens_padded
        floats = batch.floats_padded
        ints = batch.ints_padded
        uuids = batch.uuids_padded
        token_lens = (tokens != 500).sum(dim=1).tolist()
        batch_size = tokens.shape[0]
        accum_steps = max(1, TARGET_EFFECTIVE_BATCH_SIZE // batch_size)

        logits, float_preds, int_preds, uuid_preds = model(tokens[:, :-1], floats[:, :-1], ints[:, :-1], uuids[:, :-1], batch)
        
        token_targets = tokens[:, 1:]
        float_targets = floats[:, 1:]
        int_targets = ints[:, 1:]
        uuid_targets = uuids[:, 1:]

        float_mask = (token_targets == 310) & (float_targets != -1e6)
        int_mask = (token_targets == 311) & (int_targets != -1)
        uuid_mask = (token_targets == 312) & (uuid_targets != -1)

        token_loss = criterion(
            logits.reshape(-1, VOCAB_SIZE),
            token_targets.reshape(-1)
        )
        if token_loss.item() > 200:
            logits_flat = logits.reshape(-1, VOCAB_SIZE)
            token_targets_flat = token_targets.reshape(-1)
            token_loss_per_token = F.cross_entropy(logits_flat, token_targets_flat, ignore_index=token_pad_id, reduction='none')
            token_loss_per_sample = token_loss_per_token.view(tokens.shape[0], -1).mean(dim=1)
            unstable_indices = (token_loss_per_sample > 200).nonzero(as_tuple=True)[0].tolist()

            with open("Broken.txt", "a") as f:
                files = [d.mesh_file for i, d in enumerate(batch.to_data_list()) if i in unstable_indices]
                f.write(f"Files: {files}\n")
            continue
            
        float_loss = torch.tensor(0.0, device=DEVICE)
        int_loss = torch.tensor(0.0, device=DEVICE)
        uuid_loss = torch.tensor(0.0, device=DEVICE)
        
        # Normalize before computing loss
        if float_mask.any():
            norm_preds = normalize(float_preds[float_mask], mean_val, std_val)
            norm_targets = normalize(float_targets[float_mask], mean_val, std_val)
            float_loss = F.l1_loss(norm_preds, norm_targets)
        
        if int_mask.any():
            int_loss = F.cross_entropy(int_preds[int_mask], int_targets[int_mask])

        if uuid_mask.any():
            uuid_loss = F.cross_entropy(uuid_preds[uuid_mask], uuid_targets[uuid_mask])

        value_loss = float_loss_weight * float_loss + int_loss * int_loss_weight + uuid_loss * uuid_loss_weight
        
        if epoch < 11:
            value_loss_weight = 0.0
        elif epoch < 14:
            value_loss_weight = 0.3
        elif epoch < 17:
            value_loss_weight = 0.6
        elif epoch < 20:
            value_loss_weight = 0.9
        else:
            value_loss_weight = 1.0

        loss = token_loss + value_loss_weight * value_loss
        loss = loss/accum_steps

        loss.backward()
        
        if(batch_idx+1)%accum_steps==0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item()
        total_token_loss += token_loss.item()
        total_value_loss += value_loss.item()

        if (batch_idx + 1) % 10 == 0:
            progress_bar.set_postfix({
                "loss": loss.item(),
                "token_loss": token_loss.item(),
                "value_loss": value_loss.item(),
                "float_loss" : float_loss.item(),
                "int_loss" : int_loss.item(),
                "uuid_loss": uuid_loss.item(),
            })
        if batch_idx >= save_start and batch_idx % SAVE_EVERY_N_BATCHES == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch{epoch+1}_batch{batch_idx+1}.pt")
            torch.save({
                'epoch': epoch+1,
                'batch': batch_idx+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': total_loss / (batch_idx + 1)
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    avg_loss = total_loss / valid_batches
    avg_token_loss = total_token_loss / valid_batches
    avg_value_loss = total_value_loss / valid_batches

    val_loss, val_token_loss, val_value_loss, val_float_loss, val_int_loss, val_uuid_loss = evaluate(model, val_loader, criterion, DEVICE, VOCAB_SIZE, mean_val, std_val)
    print(f"Validation Token Loss: {val_token_loss:.4f}, Validation Value Loss: {val_value_loss:.4f}, validation Float Loss: {val_float_loss:.4f}, Validation Int Loss: {val_int_loss:.4f}, Validation UUID Loss: {val_uuid_loss:.4f}")
    with open("Validation.txt", "a") as f:
        f.write(
            f"Epoch {epoch+1}/{EPOCHS} - "
            f"Train Loss: {avg_loss:.4f} - "
            f"Train Token Loss: {avg_token_loss:.4f} - "
            f"Train Value Loss: {avg_value_loss:.4f} - "
            f"Val UUID Loss: {val_uuid_loss:.4f} - "
            f"Val Float Loss: {val_float_loss:.4f} - "
            f"Val Int Loss: {val_int_loss:.4f} - "
            f"Val Token Loss: {val_token_loss:.4f} - "
            f"Val Value Loss: {val_value_loss:.4f} - "
            f"Val Loss: {val_loss:.4f}\n"
        )

    if val_loss < best_val_loss or 11 <= epoch <= 20:
        best_val_loss = val_loss
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_model_epoch{epoch+1}.pt")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")



print("Training complete!")