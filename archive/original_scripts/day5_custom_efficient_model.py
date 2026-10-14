"""
Day 5: Custom Efficient Model Design
QuickDraw Sketch Recognition Project

Goal: Design a lightweight custom model that:
1. Has fewer parameters than ResNet18 (< 11.7M)
2. Achieves better accuracy than 83.49%
3. Optimized specifically for sketch recognition

Strategy:
- Depthwise Separable Convolutions (like MobileNet)
- Squeeze-and-Excitation blocks (channel attention)
- Efficient skip connections
- Optimized for 64x64 sketch images
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import utilities (utils.py lives alongside this script)
from utils import QuickDrawDataset, set_seed
from local_config import OUTPUT_DIR as LOCAL_OUTPUT_DIR, get_device

# ============================
# CONFIGURATION
# ============================

class Config:
    OUTPUT_DIR = LOCAL_OUTPUT_DIR
    PROCESSED_DIR = os.path.join(OUTPUT_DIR, 'processed_data')
    RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results/day5')
    MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')

    # Dataset parameters
    IMG_SIZE = 64
    NUM_CLASSES = 50

    # Training parameters
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 25  # More epochs for custom model
    DEVICE = get_device()
    NUM_WORKERS = 2
    PIN_MEMORY = DEVICE == 'cuda'

    # Seed
    SEED = 42

# Create directories
os.makedirs(Config.RESULTS_DIR, exist_ok=True)

# Set seed
set_seed(Config.SEED)

print("="*80)
print("DAY 5: CUSTOM EFFICIENT MODEL FOR SKETCH RECOGNITION")
print("="*80)
print(f"\nUsing device: {Config.DEVICE}")
print(f"Goal: Beat ResNet18 (83.49%) with < 11.7M parameters")

# ============================
# CUSTOM MODEL ARCHITECTURE
# ============================
#
# The SketchNet architecture itself lives in sketchnet_model.py so other
# scripts (e.g. master_pipeline_v2.py) can import the model class without
# triggering this script's training pipeline as an import side effect.

from sketchnet_model import SqueezeExcitation, DepthwiseSeparableConv, EfficientSketchBlock, SketchNet

# ============================
# LOAD DATA
# ============================

print("\n" + "="*80)
print("STEP 1: Loading Processed Data")
print("="*80)

train_df = pd.read_csv(os.path.join(Config.PROCESSED_DIR, 'train.csv'))
val_df = pd.read_csv(os.path.join(Config.PROCESSED_DIR, 'val.csv'))
test_df = pd.read_csv(os.path.join(Config.PROCESSED_DIR, 'test.csv'))

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ============================
# PREPARE DATALOADERS
# ============================

print("\n" + "="*80)
print("STEP 2: Preparing DataLoaders with Enhanced Augmentation")
print("="*80)

# Enhanced augmentation for custom model
train_transform = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
    transforms.RandomHorizontalFlip(0.3),
    transforms.RandomApply([transforms.ElasticTransform(alpha=50.0, sigma=5.0)], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = QuickDrawDataset(train_df, transform=train_transform)
val_dataset = QuickDrawDataset(val_df, transform=val_transform)
test_dataset = QuickDrawDataset(test_df, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                          shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                        shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                         shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)

print("DataLoaders ready with enhanced augmentation!")

# ============================
# CREATE AND ANALYZE MODEL
# ============================

print("\n" + "="*80)
print("STEP 3: Creating Custom SketchNet Model")
print("="*80)

model = SketchNet(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n✨ SketchNet Architecture:")
print(f"   Total Parameters: {total_params:,}")
print(f"   Trainable Parameters: {trainable_params:,}")
print(f"   Model Size: {total_params * 4 / (1024**2):.2f} MB")
print(f"\n📊 Comparison with ResNet18:")
print(f"   ResNet18: 11,689,512 parameters")
print(f"   SketchNet: {total_params:,} parameters")
print(f"   Reduction: {((11689512 - total_params) / 11689512 * 100):.1f}%")
print(f"   Memory Saved: {(11689512 - total_params) * 4 / (1024**2):.2f} MB")

# Print model architecture
print("\n📐 Model Architecture:")
print(model)

# ============================
# TRAINING SETUP
# ============================

print("\n" + "="*80)
print("STEP 4: Training Setup")
print("="*80)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

print("Optimizer: AdamW with cosine annealing")
print(f"Initial Learning Rate: {Config.LEARNING_RATE}")
print(f"Weight Decay: 0.01")
print(f"Scheduler: CosineAnnealingWarmRestarts")

# ============================
# TRAINING LOOP
# ============================

print("\n" + "="*80)
print("STEP 5: Training SketchNet")
print("="*80)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                         'acc': f'{100.*correct/total:.2f}%'})
    
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation', leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total

# Training history
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [],
    'lr': []
}

best_val_acc = 0.0
best_epoch = 0
patience = 7
patience_counter = 0

start_time = time.time()

for epoch in range(Config.NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
    print("-" * 60)
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
    
    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion, Config.DEVICE)
    
    # Update scheduler
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['lr'].append(current_lr)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    print(f"Learning Rate: {current_lr:.6f}")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'train_acc': train_acc
        }, os.path.join(Config.MODELS_DIR, 'sketchnet_best.pth'))
        print(f"✓ Best model saved! Val Acc: {val_acc:.2f}%")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping triggered after {patience} epochs without improvement")
            break

training_time = time.time() - start_time
print(f"\n✓ Training completed in {training_time/60:.2f} minutes")
print(f"✓ Best validation accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")

# ============================
# LOAD BEST MODEL AND EVALUATE
# ============================

print("\n" + "="*80)
print("STEP 6: Evaluating on Test Set")
print("="*80)

# Load best model
checkpoint = torch.load(os.path.join(Config.MODELS_DIR, 'sketchnet_best.pth'))
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluate
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc='Testing'):
        images = images.to(Config.DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

test_acc = accuracy_score(all_labels, all_preds) * 100
test_f1 = f1_score(all_labels, all_preds, average='weighted') * 100

print(f"\n{'='*80}")
print("FINAL RESULTS")
print(f"{'='*80}")
print(f"\n🎯 SketchNet Performance:")
print(f"   Test Accuracy: {test_acc:.2f}%")
print(f"   Test F1-Score: {test_f1:.2f}%")
print(f"   Best Val Accuracy: {best_val_acc:.2f}%")
print(f"   Training Time: {training_time/60:.2f} minutes")

print(f"\n📊 Comparison with ResNet18 Baseline:")
resnet18_acc = 83.49
print(f"   ResNet18: {resnet18_acc:.2f}%")
print(f"   SketchNet: {test_acc:.2f}%")
improvement = test_acc - resnet18_acc
print(f"   Improvement: {improvement:+.2f}%")

if test_acc > resnet18_acc:
    print(f"\n🎉 SUCCESS! SketchNet beats ResNet18 by {improvement:.2f}%")
    print(f"   With {((11689512 - total_params) / 11689512 * 100):.1f}% fewer parameters!")
else:
    print(f"\n📈 SketchNet: {test_acc:.2f}% (close to ResNet18: {resnet18_acc:.2f}%)")
    print(f"   But with {((11689512 - total_params) / 11689512 * 100):.1f}% fewer parameters!")

# ============================
# VISUALIZATION
# ============================

print("\n" + "="*80)
print("STEP 7: Generating Visualizations")
print("="*80)

# Plot 1: Training History
print("\n📊 Visualization 1: Training History")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Loss curves
ax1.plot(history['train_loss'], label='Train Loss', linewidth=2, color='#3498db')
ax1.plot(history['val_loss'], label='Val Loss', linewidth=2, color='#e74c3c')
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

# Accuracy curves
ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2, color='#3498db')
ax2.plot(history['val_acc'], label='Val Accuracy', linewidth=2, color='#e74c3c')
ax2.axhline(y=resnet18_acc, color='#2ecc71', linestyle='--', linewidth=2, label=f'ResNet18 ({resnet18_acc:.2f}%)')
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)

# Learning rate schedule
ax3.plot(history['lr'], linewidth=2, color='#9b59b6')
ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax3.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
ax3.grid(alpha=0.3)

# Model comparison
models_comp = ['ResNet18\n(11.7M params)', 'SketchNet\n({:.1f}M params)'.format(total_params/1e6)]
accs_comp = [resnet18_acc, test_acc]
colors_comp = ['#95a5a6', '#2ecc71' if test_acc > resnet18_acc else '#3498db']

bars = ax4.bar(models_comp, accs_comp, color=colors_comp, alpha=0.8, edgecolor='black', linewidth=2)
for bar, acc in zip(bars, accs_comp):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax4.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax4.set_title('Model Comparison', fontsize=14, fontweight='bold')
ax4.set_ylim(0, max(accs_comp) + 5)
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(Config.RESULTS_DIR, 'sketchnet_training.png'), dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved and displayed: sketchnet_training.png")

# Plot 2: Parameter Efficiency
print("\n📊 Visualization 2: Parameter Efficiency Analysis")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Parameters comparison
models = ['ResNet18', 'SketchNet']
params = [11689512, total_params]
colors = ['#95a5a6', '#2ecc71']

bars = ax1.bar(models, [p/1e6 for p in params], color=colors, alpha=0.8, edgecolor='black', linewidth=2)
for bar, p in zip(bars, params):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.3,
            f'{p/1e6:.2f}M', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax1.set_ylabel('Parameters (Millions)', fontsize=12, fontweight='bold')
ax1.set_title('Model Parameters Comparison', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Efficiency (Accuracy per Million Parameters)
efficiency = [resnet18_acc / (11689512/1e6), test_acc / (total_params/1e6)]
bars = ax2.bar(models, efficiency, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
for bar, eff in zip(bars, efficiency):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
            f'{eff:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax2.set_ylabel('Accuracy per Million Parameters', fontsize=12, fontweight='bold')
ax2.set_title('Parameter Efficiency', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(Config.RESULTS_DIR, 'parameter_efficiency.png'), dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved and displayed: parameter_efficiency.png")

# ============================
# SAVE RESULTS
# ============================

print("\n" + "="*80)
print("STEP 8: Saving Results")
print("="*80)

results = {
    'model': 'SketchNet (Custom)',
    'architecture': {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024**2),
        'vs_resnet18_reduction': f"{((11689512 - total_params) / 11689512 * 100):.1f}%"
    },
    'training': {
        'epochs_trained': len(history['train_acc']),
        'best_epoch': best_epoch,
        'training_time_minutes': training_time / 60,
        'final_learning_rate': history['lr'][-1]
    },
    'performance': {
        'test_accuracy': test_acc,
        'test_f1_score': test_f1,
        'best_val_accuracy': best_val_acc,
        'vs_resnet18': test_acc - resnet18_acc
    },
    'history': history
}

with open(os.path.join(Config.RESULTS_DIR, 'sketchnet_results.json'), 'w') as f:
    json.dump(results, f, indent=4)

print("✓ Results saved to sketchnet_results.json")

# ============================
# FINAL SUMMARY
# ============================

print("\n" + "="*80)
print("DAY 5 COMPLETED SUCCESSFULLY!")
print("="*80)

print(f"\n🎯 SketchNet Final Metrics:")
print(f"   Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
print(f"   Test Accuracy: {test_acc:.2f}%")
print(f"   Test F1-Score: {test_f1:.2f}%")
print(f"   Training Time: {training_time/60:.2f} minutes")

print(f"\n📊 vs ResNet18:")
print(f"   Parameter Reduction: {((11689512 - total_params) / 11689512 * 100):.1f}%")
print(f"   Accuracy Change: {test_acc - resnet18_acc:+.2f}%")
print(f"   Memory Saved: {(11689512 - total_params) * 4 / (1024**2):.2f} MB")

if test_acc > resnet18_acc:
    print(f"\n🏆 MISSION ACCOMPLISHED!")
    print(f"   SketchNet achieves {test_acc:.2f}% accuracy")
    print(f"   Beating ResNet18 ({resnet18_acc:.2f}%) by {test_acc - resnet18_acc:.2f}%")
    print(f"   With {((11689512 - total_params) / 11689512 * 100):.1f}% fewer parameters!")
else:
    print(f"\n✨ EXCELLENT EFFICIENCY!")
    print(f"   SketchNet achieves {test_acc:.2f}% accuracy")
    print(f"   Only {resnet18_acc - test_acc:.2f}% below ResNet18")
    print(f"   But with {((11689512 - total_params) / 11689512 * 100):.1f}% fewer parameters!")

print(f"\n📁 Outputs:")
print(f"   Model: {Config.MODELS_DIR}/sketchnet_best.pth")
print(f"   Results: {Config.RESULTS_DIR}/sketchnet_results.json")
print(f"   Plots: {Config.RESULTS_DIR}/sketchnet_training.png")
print(f"          {Config.RESULTS_DIR}/parameter_efficiency.png")

print("\n" + "="*80)