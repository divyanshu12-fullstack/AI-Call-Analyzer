
import torch
from torch.utils.data import DataLoader, random_split
from dataset import VoiceDataset
from model import VoiceCNN, VoiceResNet
import torch.nn as nn
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load full dataset and split into train/val
full_data = VoiceDataset("data/train")
val_ratio = 0.15
val_size = int(len(full_data) * val_ratio)
train_size = len(full_data) - val_size
train_data, val_data = random_split(full_data, [train_size, val_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8)

model = VoiceResNet().to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 50
patience = 7
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_path = "models/voice_model_best.pth"

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x = x.to(device)
        y = y.float().to(device)
        preds = model(x).squeeze()
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
            preds = model(x).squeeze()
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
