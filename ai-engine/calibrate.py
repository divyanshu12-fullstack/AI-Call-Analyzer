"""
Script to calibrate the trained model using validation data.
Run after training to optimize confidence calibration.
"""
import torch
from torch.utils.data import DataLoader, random_split
from dataset import VoiceDataset
from model import VoiceResNet
from calibration import calibrate_and_save
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

def calibrate_model(train_data_dir="data/train", model_path="models/voice_model_best.pth"):
    """Calibrate model on validation set and save calibration parameters."""
    
    # Load dataset
    full_data = VoiceDataset(train_data_dir, augment=False)
    val_ratio = 0.15
    val_size = int(len(full_data) * val_ratio)
    train_size = len(full_data) - val_size
    
    train_indices, val_indices = random_split(
        range(len(full_data)),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create validation dataset
    val_data = VoiceDataset(train_data_dir, augment=False)
    val_data.files = [full_data.files[i] for i in val_indices.indices]
    val_data.labels = [full_data.labels[i] for i in val_indices.indices]
    
    val_loader = DataLoader(val_data, batch_size=8)
    
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = VoiceResNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Calibrate and save
    print(f"Calibrating model on {len(val_data)} validation samples...")
    calibrated_model, best_temp = calibrate_and_save(model, val_loader, device=device)
    
    print(f"✓ Calibration complete!")
    print(f"✓ Best temperature: {best_temp:.4f}")
    print(f"✓ Calibration saved to models/calibration.pth")

if __name__ == "__main__":
    calibrate_model()
