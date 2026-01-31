import torch
from torch.utils.data import DataLoader
from dataset import VoiceDataset
from model import VoiceResNet

device = "cuda" if torch.cuda.is_available() else "cpu"

data = VoiceDataset("data/test")
loader = DataLoader(data, batch_size=8)

model = VoiceResNet().to(device)
model.load_state_dict(torch.load("models/voice_model_best.pth"))
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

acc = correct / total
print("Accuracy:", acc)
with open("evaluation_results.txt", "w") as f:
    f.write(f"Test Accuracy: {acc*100:.2f}%\n")
    f.write(f"Correct: {correct}/{total}\n")

