
import torch
from torch.utils.data import DataLoader, random_split
from dataset import VoiceDataset
from model import VoiceResNet
import torch.nn as nn
import numpy as np
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load full dataset (without augmentation initially for splitting)
full_data = VoiceDataset("data/train", augment=False)
val_ratio = 0.15
val_size = int(len(full_data) * val_ratio)
train_size = len(full_data) - val_size
train_indices, val_indices = random_split(
    range(len(full_data)), 
    [train_size, val_size], 
    generator=torch.Generator().manual_seed(42)
)

# Create datasets with proper augmentation settings
train_data = VoiceDataset("data/train", augment=True)
train_data.files = [full_data.files[i] for i in train_indices.indices]
train_data.labels = [full_data.labels[i] for i in train_indices.indices]

val_data = VoiceDataset("data/train", augment=False)
val_data.files = [full_data.files[i] for i in val_indices.indices]
val_data.labels = [full_data.labels[i] for i in val_indices.indices]

# CRITICAL FIX: Use num_workers=0 to avoid Windows hanging issues with DataLoader
train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=8, num_workers=0)

model = VoiceResNet().to(device)

# Custom Focal Loss for better hard-example mining
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)  # pt is the probability of being right
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

criterion = FocalLoss()  # Replaces standard BCELoss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 50
patience = 7
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_path = "models/voice_model_best.pth"

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x = x.to(device)
        y = y.float().to(device)
        preds = model(x)
        y = y.view(-1, 1)  # Ensure target is (batch, 1)
        loss = criterion(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.float().to(device)
            preds = model(x)
            y = y.view(-1, 1)
            loss = criterion(preds, y)
            val_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    # Early stopping and checkpoint
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"  Best model saved at epoch {epoch+1}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Save final model
torch.save(model.state_dict(), "models/voice_model_final.pth")
print("Final model saved as models/voice_model_final.pth")
print(f"Best model saved as {best_model_path}")
