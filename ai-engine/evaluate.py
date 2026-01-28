import torch
from torch.utils.data import DataLoader
from dataset import VoiceDataset
from model import VoiceCNN

device = "cuda" if torch.cuda.is_available() else "cpu"

data = VoiceDataset("data/test")
loader = DataLoader(data, batch_size=8)

model = VoiceCNN().to(device)
model.load_state_dict(torch.load("voice_model.pth"))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        preds = model(x).squeeze()
        preds = (preds > 0.5).int()

        correct += (preds == y).sum().item()
        total += y.size(0)

print("Accuracy:", correct / total)
