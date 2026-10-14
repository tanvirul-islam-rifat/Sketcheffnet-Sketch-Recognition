"""
Day 1: Dataset Setup, EDA, and Baseline Model Training
QuickDraw Sketch Recognition Project

Runs locally (CUDA / Apple Silicon MPS / CPU auto-detected).
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import utilities (make sure utils.py is in the same directory)
from utils import (
    set_seed, prepare_dataset_from_npy, QuickDrawDataset,
    get_transforms, split_dataset, plot_samples, plot_class_distribution,
    plot_confusion_matrix, save_results, plot_training_curves
)
from local_config import DATA_DIR as LOCAL_DATA_DIR, OUTPUT_DIR as LOCAL_OUTPUT_DIR, get_device

# ============================
# CONFIGURATION
# ============================

class Config:
    # Paths (see local_config.py to change these)
    DATA_DIR = LOCAL_DATA_DIR
    OUTPUT_DIR = LOCAL_OUTPUT_DIR
    PROCESSED_DIR = os.path.join(OUTPUT_DIR, 'processed_data')
    RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results/day1')
    MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')

    # Dataset parameters
    SAMPLES_PER_CATEGORY = 2000
    IMG_SIZE = 64
    NUM_CLASSES = 50

    # Training parameters
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    DEVICE = get_device()
    NUM_WORKERS = 2
    PIN_MEMORY = DEVICE == 'cuda'

    # Seed
    SEED = 42

# Create directories
os.makedirs(Config.RESULTS_DIR, exist_ok=True)
os.makedirs(Config.MODELS_DIR, exist_ok=True)

# Set seed
set_seed(Config.SEED)

print(f"Using device: {Config.DEVICE}")
if Config.DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
elif Config.DEVICE == 'mps':
    print("GPU: Apple Silicon (MPS backend)")

# ============================
# STEP 1: LOAD AND PREPARE DATASET
# ============================

print("\n" + "="*50)
print("STEP 1: Loading and Preparing Dataset")
print("="*50)

# Define 50 categories (adjust these to match your actual files)
CATEGORIES = [
    'aircraft carrier', 'airplane', 'alarm clock', 'ambulance', 'angel',
    'ant', 'apple', 'arm', 'asparagus', 'axe',
    'banana', 'bandage', 'barn', 'baseball', 'baseball bat',
    'basket', 'basketball', 'bat', 'bathtub', 'beach',
    'bear', 'beard', 'bed', 'bee', 'belt',
    'bench', 'bicycle', 'binoculars', 'bird', 'birthday cake',
    'blackberry', 'blueberry', 'book', 'boomerang', 'bottlecap',
    'bowtie', 'bracelet', 'brain', 'bread', 'bridge',
    'broccoli', 'broom', 'bucket', 'bulldozer', 'bus',
    'bush', 'butterfly', 'cactus', 'cake', 'calculator'
]

# Check if processed data already exists
dataset_csv = os.path.join(Config.PROCESSED_DIR, 'dataset.csv')

if os.path.exists(dataset_csv):
    print("Loading existing processed dataset...")
    df = pd.read_csv(dataset_csv)
    print(f"Loaded {len(df)} images")
else:
    print("Processing dataset from .npy files...")
    print(f"Looking for files in: {Config.DATA_DIR}")
    
    # List available files
    if os.path.exists(Config.DATA_DIR):
        files = [f for f in os.listdir(Config.DATA_DIR) if f.endswith('.npy')]
        print(f"Found {len(files)} .npy files in data directory")
        if len(files) > 0:
            print(f"Sample files: {files[:5]}")
        else:
            print("⚠️  WARNING: No .npy files found!")
            print("Please check your DATA_DIR path.")
    
    df = prepare_dataset_from_npy(
        data_dir=Config.DATA_DIR,
        categories=CATEGORIES,
        samples_per_category=Config.SAMPLES_PER_CATEGORY,
        output_dir=Config.PROCESSED_DIR,
        img_size=Config.IMG_SIZE
    )

print(f"\nDataset shape: {df.shape}")
print(f"Categories: {df['category'].nunique()}")
print(f"Samples per category:")
print(df['category'].value_counts().sort_index())

# ============================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ============================

print("\n" + "="*50)
print("STEP 2: Exploratory Data Analysis")
print("="*50)

# Plot sample images
print("Generating sample images...")
plot_samples(df, CATEGORIES, n_samples=5, 
             save_path=os.path.join(Config.RESULTS_DIR, 'sample_images.png'))

# Plot class distribution
print("Generating class distribution plot...")
plot_class_distribution(df, 
                        save_path=os.path.join(Config.RESULTS_DIR, 'class_distribution.png'))

# Basic statistics
print("\nDataset Statistics:")
print(f"Total images: {len(df)}")
print(f"Number of categories: {df['category'].nunique()}")
print(f"Images per category (mean): {df['category'].value_counts().mean():.2f}")
print(f"Images per category (std): {df['category'].value_counts().std():.2f}")

# ============================
# STEP 3: SPLIT DATASET
# ============================

print("\n" + "="*50)
print("STEP 3: Splitting Dataset")
print("="*50)

train_df, val_df, test_df = split_dataset(df, train_ratio=0.7, val_ratio=0.15, seed=Config.SEED)

print(f"Train set: {len(train_df)} images")
print(f"Validation set: {len(val_df)} images")
print(f"Test set: {len(test_df)} images")

# Save splits
train_df.to_csv(os.path.join(Config.PROCESSED_DIR, 'train.csv'), index=False)
val_df.to_csv(os.path.join(Config.PROCESSED_DIR, 'val.csv'), index=False)
test_df.to_csv(os.path.join(Config.PROCESSED_DIR, 'test.csv'), index=False)

# ============================
# STEP 4: CREATE DATALOADERS
# ============================

print("\n" + "="*50)
print("STEP 4: Creating DataLoaders")
print("="*50)

train_transform, val_transform = get_transforms(img_size=Config.IMG_SIZE, augment=True)

train_dataset = QuickDrawDataset(train_df, transform=train_transform)
val_dataset = QuickDrawDataset(val_df, transform=val_transform)
test_dataset = QuickDrawDataset(test_df, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                          shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                        shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                         shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)

print(f"Train batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# ============================
# STEP 5: DEFINE BASELINE MODEL
# ============================

print("\n" + "="*50)
print("STEP 5: Defining Baseline Model (ResNet18)")
print("="*50)

def create_model(num_classes, pretrained=True):
    """Create ResNet18 model"""
    model = models.resnet18(pretrained=pretrained)
    
    # Modify final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

model = create_model(Config.NUM_CLASSES, pretrained=True)
model = model.to(Config.DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                                  patience=3)

print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

# ============================
# STEP 6: TRAINING FUNCTIONS
# ============================

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
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
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    """Validate the model"""
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
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                             'acc': f'{100.*correct/total:.2f}%'})
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

# ============================
# STEP 7: TRAIN BASELINE MODEL
# ============================

print("\n" + "="*50)
print("STEP 7: Training Baseline Model")
print("="*50)

history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

best_val_acc = 0.0
start_time = time.time()

for epoch in range(Config.NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
    val_loss, val_acc = validate(model, val_loader, criterion, Config.DEVICE)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Learning rate scheduling
    scheduler.step(val_acc)
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, os.path.join(Config.MODELS_DIR, 'baseline_best.pth'))
        print(f"✓ Best model saved with val_acc: {val_acc:.2f}%")

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time/60:.2f} minutes")
print(f"Best validation accuracy: {best_val_acc:.2f}%")

# Plot training curves
plot_training_curves(history, save_path=os.path.join(Config.RESULTS_DIR, 'training_curves.png'))

# ============================
# STEP 8: EVALUATE ON TEST SET
# ============================

print("\n" + "="*50)
print("STEP 8: Evaluating on Test Set")
print("="*50)

# Load best model
checkpoint = torch.load(os.path.join(Config.MODELS_DIR, 'baseline_best.pth'))
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

# Calculate metrics
test_acc = accuracy_score(all_labels, all_preds) * 100
test_f1 = f1_score(all_labels, all_preds, average='weighted') * 100

print(f"\nTest Accuracy: {test_acc:.2f}%")
print(f"Test F1-Score: {test_f1:.2f}%")

# Classification report
print("\nClassification Report:")
report = classification_report(all_labels, all_preds, target_names=CATEGORIES, 
                              digits=4, zero_division=0)
print(report)

# Save classification report
with open(os.path.join(Config.RESULTS_DIR, 'classification_report.txt'), 'w') as f:
    f.write(report)

# Plot confusion matrix
plot_confusion_matrix(all_labels, all_preds, CATEGORIES,
                     save_path=os.path.join(Config.RESULTS_DIR, 'confusion_matrix.png'))

# ============================
# STEP 9: SAVE RESULTS
# ============================

print("\n" + "="*50)
print("STEP 9: Saving Results")
print("="*50)

results = {
    'model': 'ResNet18 (Baseline)',
    'dataset': {
        'total_samples': len(df),
        'samples_per_category': Config.SAMPLES_PER_CATEGORY,
        'num_categories': Config.NUM_CLASSES,
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df)
    },
    'training': {
        'epochs': Config.NUM_EPOCHS,
        'batch_size': Config.BATCH_SIZE,
        'learning_rate': Config.LEARNING_RATE,
        'training_time_minutes': training_time/60,
        'best_val_acc': best_val_acc
    },
    'test_metrics': {
        'accuracy': test_acc,
        'f1_score': test_f1
    },
    'history': history
}

save_results(results, os.path.join(Config.RESULTS_DIR, 'baseline_results.json'))

print("\n" + "="*50)
print("DAY 1 COMPLETED SUCCESSFULLY!")
print("="*50)
print(f"\nResults saved to: {Config.RESULTS_DIR}")
print(f"Model saved to: {Config.MODELS_DIR}")
print(f"\nBaseline Test Accuracy: {test_acc:.2f}%")
print(f"Baseline Test F1-Score: {test_f1:.2f}%")