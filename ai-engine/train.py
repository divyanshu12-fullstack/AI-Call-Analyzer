import torch
from torch.utils.data import DataLoader
from dataset import VoiceDataset
from model import VoiceCNN
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

train_data = VoiceDataset("data/train")
test_data = VoiceDataset("data/test")

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=8)

model = VoiceCNN().to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 20

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

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "voice_model.pth")
print("Model saved as voice_model.pth")
