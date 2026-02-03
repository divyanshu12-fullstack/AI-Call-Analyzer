"""
Confidence calibration for neural networks.
Implements temperature scaling for post-hoc calibration.
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import os

class TemperatureScaling(nn.Module):
    """Temperature scaling for calibration."""
    def __init__(self, temperature=1.0):
        super().__init__()
        self.register_buffer('temperature', torch.tensor(temperature))
    
    def forward(self, logits):
        return logits / self.temperature
    
    def save(self, path):
        """Save temperature parameter."""
        torch.save({'temperature': self.temperature.item()}, path)
    
    def load(self, path):
        """Load temperature parameter."""
        data = torch.load(path, weights_only=True)
        self.temperature.fill_(data['temperature'])


def calibrate_temperature(model, val_loader, device='cpu'):
    """
    Find optimal temperature via grid search on validation set.
    Returns calibrated model and best temperature.
    """
    model.eval()
    
    # Collect predictions
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.cpu().numpy()
            
            probs = model(x).squeeze().cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y)
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Grid search for best temperature
    best_temp = 1.0
    best_ece = float('inf')
    
    temperatures = np.linspace(0.1, 5.0, 50)
    
    for temp in temperatures:
        # Apply temperature scaling
        calibrated_probs = 1.0 / (1.0 + np.exp(-(np.log(all_probs / (1 - all_probs + 1e-10)) / temp)))
        calibrated_probs = np.clip(calibrated_probs, 1e-15, 1 - 1e-15)
        
        # Compute ECE (Expected Calibration Error)
        ece = compute_ece(all_labels, calibrated_probs, n_bins=10)
        
        if ece < best_ece:
            best_ece = ece
            best_temp = temp
    
    print(f"Optimal temperature: {best_temp:.4f}")
    print(f"ECE with calibration: {best_ece:.4f}")
    
    # Create calibrated model
    calibrated_model = TemperatureScaling(temperature=best_temp)
    
    return calibrated_model, best_temp


def compute_ece(labels, probs, n_bins=10):
    """
    Compute Expected Calibration Error.
    Lower ECE indicates better calibration.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    total_samples = len(labels)
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.sum() / total_samples
        
        if prop_in_bin > 0:
            accuracy_in_bin = (labels[in_bin] == (probs[in_bin] > 0.5)).astype(float).mean()
            avg_confidence_in_bin = probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def calibrate_and_save(model, val_loader, device='cpu', save_dir='models'):
    """
    Calibrate model and save calibration parameters.
    """
    calibrated_model, best_temp = calibrate_temperature(model, val_loader, device)
    
    os.makedirs(save_dir, exist_ok=True)
    calibration_path = os.path.join(save_dir, 'calibration.pth')
    calibrated_model.save(calibration_path)
    
    print(f"Calibration saved to {calibration_path}")
    
    return calibrated_model, best_temp
