import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from MLDataset import MeshCADTokenDataset
from Encode_Decode import MeshToCADModel
from utils import collate_fn
from tqdm import tqdm
import os

BATCH_SIZE = 4
EPOCHS = 10
DEVICE = torch.device('cuda')
EOS_TOKEN = 501
VOCAB_SIZE = 502
SAVE_EVERY_N_BATCHES = 100
CHECKPOINT_DIR = "MeshtoCAD/ML/checkpoints"
RESUME_PATH =  None
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

train_dataset = MeshCADTokenDataset("MeshtoCAD/TokenizedJSON/train")
val_dataset = MeshCADTokenDataset("MeshtoCAD/TokenizedJSON/val")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

model = MeshToCADModel(
    mesh_in_channels=3,
    mesh_hidden=128,
    mesh_latent=256,
    vocab_size=VOCAB_SIZE,
    dec_hidden=256,
    dec_layers=2
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=500) 

def evaluate(model, loader, criterion, device, vocab_size):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
              continue
            batch = batch.to(device)
            tokens = batch.tokens_padded
            values = batch.values_padded
            logits = model(batch, tokens[:, :-1], values[:, :-1])
            loss = criterion(
                logits.reshape(-1, VOCAB_SIZE),
                tokens[:, 1:].reshape(-1)
            )
            total_loss += loss.item()
    return total_loss / len(loader)

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

    val_loss = evaluate(model, val_loader, criterion, DEVICE, VOCAB_SIZE)
    print(f"Validation Loss after resume: {val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_model_epoch{start_epoch}_resume.pt")
        torch.save({
            'epoch': start_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
else:
    print("Starting training from scratch.")

for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch_idx, batch in enumerate(progress_bar):
        if batch is None:
            continue
        batch = batch.to(DEVICE)
        tokens = getattr(batch, 'tokens_padded', None)
        values = getattr(batch, 'values_padded', None)
        if tokens is None or tokens.dim() != 2:
            raise RuntimeError(f"Batch tokens_padded shape is invalid: {tokens}")
        logits = model(batch, tokens[:, :-1], values[:, :-1])
        print(
            f"Batch tokens: min={tokens.min().item()}, max={tokens.max().item()}, mean={tokens.float().mean().item()}"
        )
        print(
            f"Batch values: min={values.min().item()}, max={values.max().item()}, mean={values.float().mean().item()}"
        )
        loss = criterion(
            logits.reshape(-1, VOCAB_SIZE),
            tokens[:, 1:].reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:
            progress_bar.set_postfix({"batch_loss": loss.item()})

        if (batch_idx + 1) % SAVE_EVERY_N_BATCHES == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch{epoch+1}_batch{batch_idx+1}.pt")
            torch.save({
                'epoch': epoch+1,
                'batch': batch_idx+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': total_loss / (batch_idx + 1)
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    avg_loss = total_loss / len(train_loader)
    val_loss = evaluate(model, val_loader, criterion, DEVICE, VOCAB_SIZE)
    with open("Validation.txt", "a") as f:
        f.write(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f}\n")
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f}")


    if val_loss < best_val_loss:
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