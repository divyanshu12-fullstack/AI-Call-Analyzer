"""
Comprehensive evaluation script for voice detection model.
Computes accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix.
"""
import torch
from torch.utils.data import DataLoader
from dataset import VoiceDataset
from model import VoiceResNet
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_model(model_path="models/voice_model_best.pth", data_dir="data/test"):
    """Evaluate model and compute comprehensive metrics."""
    
    # Load test data
    test_data = VoiceDataset(data_dir, augment=False)
    test_loader = DataLoader(test_data, batch_size=8)
    
    # Load model
    model = VoiceResNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Collect predictions and ground truth
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.cpu().numpy()
            
            probs = model(x).squeeze().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y)
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs)
    
    # Per-class metrics
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    # Confusion matrix with explicit labels to ensure 2x2 matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    
    # False positives and false negatives
    if cm.size == 4:  # Proper 2x2 confusion matrix
        tn, fp, fn, tp = cm.ravel()
    else:
        # Fallback if confusion matrix is not 2x2
        tn, fp, fn, tp = 0, 0, 0, 0
        print("Warning: Confusion matrix is not 2x2, setting FP/FN rates to 0")
    
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    
    # Print results
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*60)
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    print(f"\nPer-Class Metrics:")
    # Handle cases where only one class is predicted
    if len(precision_per_class) > 0:
        print(f"  Human (0):        Precision={precision_per_class[0]:.4f}, Recall={recall_per_class[0]:.4f}, F1={f1_per_class[0]:.4f}")
    if len(precision_per_class) > 1:
        print(f"  AI-Generated (1): Precision={precision_per_class[1]:.4f}, Recall={recall_per_class[1]:.4f}, F1={f1_per_class[1]:.4f}")
    else:
        print(f"  AI-Generated (1): No predictions made for this class")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives:  {tp}")
    
    print(f"\nError Analysis:")
    print(f"  False Positive Rate: {false_positive_rate:.4f}")
    print(f"  False Negative Rate: {false_negative_rate:.4f}")
    
    # Save detailed results to file
    with open("evaluation_results.txt", "w") as f:
        f.write("="*60 + "\n")
        f.write("COMPREHENSIVE EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Test Samples: {len(all_labels)}\n\n")
        
        f.write("Overall Metrics:\n")
        f.write(f"  Accuracy:  {accuracy:.4f}\n")
        f.write(f"  Precision: {precision:.4f}\n")
        f.write(f"  Recall:    {recall:.4f}\n")
        f.write(f"  F1-Score:  {f1:.4f}\n")
        f.write(f"  ROC-AUC:   {roc_auc:.4f}\n\n")
        
        f.write("Per-Class Metrics:\n")
        if len(precision_per_class) > 0:
            f.write(f"  Human (0):        Precision={precision_per_class[0]:.4f}, Recall={recall_per_class[0]:.4f}, F1={f1_per_class[0]:.4f}\n")
        if len(precision_per_class) > 1:
            f.write(f"  AI-Generated (1): Precision={precision_per_class[1]:.4f}, Recall={recall_per_class[1]:.4f}, F1={f1_per_class[1]:.4f}\n\n")
        else:
            f.write(f"  AI-Generated (1): No predictions made for this class\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write(f"  True Negatives:  {tn}\n")
        f.write(f"  False Positives: {fp}\n")
        f.write(f"  False Negatives: {fn}\n")
        f.write(f"  True Positives:  {tp}\n\n")
        
        f.write("Error Analysis:\n")
        f.write(f"  False Positive Rate: {false_positive_rate:.4f}\n")
        f.write(f"  False Negative Rate: {false_negative_rate:.4f}\n")
    
    print(f"\nResults saved to evaluation_results.txt")
    
    # Create visualizations
    os.makedirs("plots", exist_ok=True)
    
    # 1. Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Human', 'AI'], 
                yticklabels=['Human', 'AI'],
                cbar_kws={'label': 'Count'})
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png', dpi=100)
    print(f"Confusion matrix saved to plots/confusion_matrix.png")
    plt.close()
    
    # 2. ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/roc_curve.png', dpi=100)
    print(f"ROC curve saved to plots/roc_curve.png")
    plt.close()
    
    # 3. Metrics Comparison
    metrics_dict = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    }
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics_dict.keys(), metrics_dict.values(), color='steelblue', alpha=0.7)
    plt.ylabel('Score')
    plt.title('Overall Metrics Comparison')
    plt.ylim([0, 1.0])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/metrics_comparison.png', dpi=100)
    print(f"Metrics comparison saved to plots/metrics_comparison.png")
    plt.close()
    
    # 4. Per-Class Metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    classes = ['Human', 'AI-Generated']
    
    for idx, (ax, metric_arr, title) in enumerate([
        (axes[0], precision_per_class, 'Precision'),
        (axes[1], recall_per_class, 'Recall'),
        (axes[2], f1_per_class, 'F1-Score')
    ]):
        bars = ax.bar(classes, metric_arr, color=['green', 'red'], alpha=0.7)
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_ylim([0, 1.0])
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/per_class_metrics.png', dpi=100)
    print(f"Per-class metrics saved to plots/per_class_metrics.png")
    plt.close()
    
    print("="*60)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate
    }

if __name__ == "__main__":
    results = evaluate_model()

