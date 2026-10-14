"""
Day 2: Core Experiments
1. Few-Shot Learning Simulation
2. Sketch-Specific Data Augmentation
3. Transfer Learning Analysis

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
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score
from sklearn.manifold import TSNE
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import utilities
from utils import (
    set_seed, QuickDrawDataset, get_transforms, save_results
)
from local_config import OUTPUT_DIR as LOCAL_OUTPUT_DIR, get_device

# ============================
# CONFIGURATION
# ============================

class Config:
    # Paths
    OUTPUT_DIR = LOCAL_OUTPUT_DIR
    PROCESSED_DIR = os.path.join(OUTPUT_DIR, 'processed_data')
    RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results/day2')
    MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')

    # Dataset parameters
    IMG_SIZE = 64
    NUM_CLASSES = 50

    # Training parameters
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    NUM_EPOCHS_FEWSHOT = 15
    NUM_EPOCHS_AUG = 15
    NUM_EPOCHS_TRANSFER = 15
    DEVICE = get_device()
    NUM_WORKERS = 2

    # Few-shot settings
    FEW_SHOT_SAMPLES = [10, 50, 100, 500, 1000]

    # Transfer learning settings
    TRANSFER_TRAIN_CATEGORIES = 40

    # Seed
    SEED = 42

# Create directories
os.makedirs(Config.RESULTS_DIR, exist_ok=True)

# Set seed
set_seed(Config.SEED)

print(f"Using device: {Config.DEVICE}")

# ============================
# LOAD PROCESSED DATA
# ============================

print("\n" + "="*50)
print("Loading Processed Dataset")
print("="*50)

train_df = pd.read_csv(os.path.join(Config.PROCESSED_DIR, 'train.csv'))
val_df = pd.read_csv(os.path.join(Config.PROCESSED_DIR, 'val.csv'))
test_df = pd.read_csv(os.path.join(Config.PROCESSED_DIR, 'test.csv'))

CATEGORIES = sorted(train_df['category'].unique())

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
print(f"Categories: {len(CATEGORIES)}")

# ============================
# HELPER FUNCTIONS
# ============================

def create_model(num_classes, pretrained=True):
    """Create ResNet18 model"""
    model = models.resnet18(pretrained=pretrained)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, device, model_name='model'):
    """Train model and return best validation accuracy"""
    best_val_acc = 0.0
    history = {'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        # Train
        model.train()
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        history['train_acc'].append(train_acc)

        # Validate
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = 100. * correct / total
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    return best_val_acc, history

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average='weighted') * 100

    return acc, f1

# ============================
# EXPERIMENT 1: FEW-SHOT LEARNING
# ============================

print("\n" + "="*60)
print("EXPERIMENT 1: Few-Shot Learning Simulation")
print("="*60)

few_shot_results = {
    'samples': [],
    'train_acc': [],
    'val_acc': [],
    'test_acc': [],
    'test_f1': []
}

_, val_transform = get_transforms(img_size=Config.IMG_SIZE, augment=False)
val_dataset = QuickDrawDataset(val_df, transform=val_transform)
test_dataset = QuickDrawDataset(test_df, transform=val_transform)

val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                        shuffle=False, num_workers=Config.NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                         shuffle=False, num_workers=Config.NUM_WORKERS)

for n_samples in Config.FEW_SHOT_SAMPLES:
    print(f"\n--- Training with {n_samples} samples per category ---")

    # Create few-shot subset
    few_shot_dfs = []
    for category in CATEGORIES:
        category_df = train_df[train_df['category'] == category]
        sampled = category_df.sample(n=min(n_samples, len(category_df)), random_state=Config.SEED)
        few_shot_dfs.append(sampled)

    few_shot_train_df = pd.concat(few_shot_dfs, ignore_index=True)
    print(f"Few-shot training samples: {len(few_shot_train_df)}")

    # Create dataloader
    train_transform, _ = get_transforms(img_size=Config.IMG_SIZE, augment=True)
    train_dataset = QuickDrawDataset(few_shot_train_df, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                              shuffle=True, num_workers=Config.NUM_WORKERS)

    # Train model
    model = create_model(Config.NUM_CLASSES, pretrained=True).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    best_val_acc, history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        Config.NUM_EPOCHS_FEWSHOT, Config.DEVICE, f'fewshot_{n_samples}'
    )

    # Evaluate
    test_acc, test_f1 = evaluate_model(model, test_loader, Config.DEVICE)

    few_shot_results['samples'].append(n_samples)
    few_shot_results['train_acc'].append(history['train_acc'][-1])
    few_shot_results['val_acc'].append(best_val_acc)
    few_shot_results['test_acc'].append(test_acc)
    few_shot_results['test_f1'].append(test_f1)

    print(f"Results - Val Acc: {best_val_acc:.2f}%, Test Acc: {test_acc:.2f}%, Test F1: {test_f1:.2f}%")

# Plot few-shot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(few_shot_results['samples'], few_shot_results['test_acc'], marker='o', linewidth=2, markersize=8)
ax1.set_xlabel('Samples per Category', fontsize=12)
ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
ax1.set_title('Few-Shot Learning: Accuracy vs Training Samples', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')

ax2.plot(few_shot_results['samples'], few_shot_results['test_f1'], marker='s', 
         linewidth=2, markersize=8, color='orange')
ax2.set_xlabel('Samples per Category', fontsize=12)
ax2.set_ylabel('Test F1-Score (%)', fontsize=12)
ax2.set_title('Few-Shot Learning: F1-Score vs Training Samples', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')

plt.tight_layout()
plt.savefig(os.path.join(Config.RESULTS_DIR, 'fewshot_results.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\nFew-shot learning results saved!")

# ============================
# EXPERIMENT 2: AUGMENTATION STRATEGIES
# ============================

print("\n" + "="*60)
print("EXPERIMENT 2: Sketch-Specific Data Augmentation")
print("="*60)

# Define augmentation strategies
augmentation_configs = {
    'no_augmentation': transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'standard_augmentation': transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'sketch_specific_augmentation': transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomApply([
            transforms.ElasticTransform(alpha=50.0, sigma=5.0)
        ], p=0.3),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

augmentation_results = {
    'strategy': [],
    'val_acc': [],
    'test_acc': [],
    'test_f1': []
}

# Use subset for faster training (500 samples per category)
subset_dfs = []
for category in CATEGORIES:
    category_df = train_df[train_df['category'] == category]
    sampled = category_df.sample(n=min(500, len(category_df)), random_state=Config.SEED)
    subset_dfs.append(sampled)
subset_train_df = pd.concat(subset_dfs, ignore_index=True)

for aug_name, aug_transform in augmentation_configs.items():
    print(f"\n--- Training with {aug_name} ---")

    # Create dataloaders
    train_dataset = QuickDrawDataset(subset_train_df, transform=aug_transform)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                              shuffle=True, num_workers=Config.NUM_WORKERS)

    # Train model
    model = create_model(Config.NUM_CLASSES, pretrained=True).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    best_val_acc, history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        Config.NUM_EPOCHS_AUG, Config.DEVICE, aug_name
    )

    # Evaluate
    test_acc, test_f1 = evaluate_model(model, test_loader, Config.DEVICE)

    augmentation_results['strategy'].append(aug_name)
    augmentation_results['val_acc'].append(best_val_acc)
    augmentation_results['test_acc'].append(test_acc)
    augmentation_results['test_f1'].append(test_f1)

    print(f"Results - Val Acc: {best_val_acc:.2f}%, Test Acc: {test_acc:.2f}%, Test F1: {test_f1:.2f}%")

# Plot augmentation results
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(augmentation_results['strategy']))
width = 0.35

bars1 = ax.bar(x - width/2, augmentation_results['test_acc'], width, label='Test Accuracy', alpha=0.8)
bars2 = ax.bar(x + width/2, augmentation_results['test_f1'], width, label='Test F1-Score', alpha=0.8)

ax.set_xlabel('Augmentation Strategy', fontsize=12)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Impact of Different Augmentation Strategies', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels([s.replace('_', ' ').title() for s in augmentation_results['strategy']], rotation=15, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(Config.RESULTS_DIR, 'augmentation_results.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\nAugmentation comparison results saved!")

# ============================
# EXPERIMENT 3: TRANSFER LEARNING
# ============================

print("\n" + "="*60)
print("EXPERIMENT 3: Transfer Learning Analysis")
print("="*60)

# Split categories: 40 for training, 10 for testing
np.random.seed(Config.SEED)
all_categories = np.array(CATEGORIES)
np.random.shuffle(all_categories)

train_categories = all_categories[:Config.TRANSFER_TRAIN_CATEGORIES].tolist()
test_categories = all_categories[Config.TRANSFER_TRAIN_CATEGORIES:].tolist()

print(f"Train categories ({len(train_categories)}): {train_categories[:5]}...")
print(f"Test categories ({len(test_categories)}): {test_categories}")

# Create category-specific datasets
transfer_train_df = train_df[train_df['category'].isin(train_categories)].copy()
transfer_val_df = val_df[val_df['category'].isin(train_categories)].copy()
transfer_test_df = test_df[test_df['category'].isin(test_categories)].copy()

# Remap labels for training (0 to 39)
category_to_label = {cat: idx for idx, cat in enumerate(train_categories)}
transfer_train_df['label'] = transfer_train_df['category'].map(category_to_label)
transfer_val_df['label'] = transfer_val_df['category'].map(category_to_label)

print(f"\nTransfer train: {len(transfer_train_df)}")
print(f"Transfer val: {len(transfer_val_df)}")
print(f"Transfer test: {len(transfer_test_df)}")

# Train on subset of categories
train_transform, val_transform = get_transforms(img_size=Config.IMG_SIZE, augment=True)

transfer_train_dataset = QuickDrawDataset(transfer_train_df, transform=train_transform)
transfer_val_dataset = QuickDrawDataset(transfer_val_df, transform=val_transform)

transfer_train_loader = DataLoader(transfer_train_dataset, batch_size=Config.BATCH_SIZE,
                                  shuffle=True, num_workers=Config.NUM_WORKERS)
transfer_val_loader = DataLoader(transfer_val_dataset, batch_size=Config.BATCH_SIZE,
                                shuffle=False, num_workers=Config.NUM_WORKERS)

# Train model on 40 categories
print("\nTraining on 40 categories...")
transfer_model = create_model(num_classes=40, pretrained=True).to(Config.DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(transfer_model.parameters(), lr=Config.LEARNING_RATE)

best_val_acc, history = train_model(
    transfer_model, transfer_train_loader, transfer_val_loader,
    criterion, optimizer, Config.NUM_EPOCHS_TRANSFER, Config.DEVICE, 'transfer'
)

print(f"\nBest validation accuracy on 40 categories: {best_val_acc:.2f}%")

# Extract features for unseen categories
print("\nExtracting features for transfer learning analysis...")

# Modify model to extract features
feature_extractor = nn.Sequential(*list(transfer_model.children())[:-1])
feature_extractor.eval()

def extract_features(model, dataloader, device):
    """Extract features from model"""
    features = []
    labels = []

    with torch.no_grad():
        for images, lbls in tqdm(dataloader, desc='Extracting features', leave=False):
            images = images.to(device)
            feats = model(images)
            feats = feats.squeeze()

            features.append(feats.cpu().numpy())
            labels.extend(lbls.numpy())

    return np.vstack(features), np.array(labels)

# Extract features from test categories
transfer_test_dataset = QuickDrawDataset(transfer_test_df, transform=val_transform)
transfer_test_loader = DataLoader(transfer_test_dataset, batch_size=Config.BATCH_SIZE,
                                 shuffle=False, num_workers=Config.NUM_WORKERS)

test_features, test_labels = extract_features(feature_extractor, transfer_test_loader, Config.DEVICE)

print(f"Extracted features shape: {test_features.shape}")

# Analyze feature separability using k-NN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Split test features for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    test_features, test_labels, test_size=0.3, random_state=Config.SEED, stratify=test_labels
)

# Train k-NN classifier on extracted features
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_acc = knn.score(X_test, y_test) * 100

print(f"\nk-NN accuracy on unseen categories (using transferred features): {knn_acc:.2f}%")

# Visualize features using t-SNE
print("\nGenerating t-SNE visualization...")

# Sample for faster t-SNE
n_samples = min(1000, len(test_features))
indices = np.random.choice(len(test_features), n_samples, replace=False)
sampled_features = test_features[indices]
sampled_labels = test_labels[indices]

tsne = TSNE(n_components=2, random_state=Config.SEED, perplexity=30)
features_2d = tsne.fit_transform(sampled_features)

plt.figure(figsize=(12, 10))
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                     c=sampled_labels, cmap='tab10', alpha=0.6, s=20)
plt.colorbar(scatter, label='Category')
plt.title('t-SNE Visualization of Transferred Features (Unseen Categories)', fontsize=14)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.tight_layout()
plt.savefig(os.path.join(Config.RESULTS_DIR, 'transfer_tsne.png'), dpi=150, bbox_inches='tight')
plt.close()

print("t-SNE visualization saved!")

transfer_results = {
    'train_categories': train_categories,
    'test_categories': test_categories,
    'best_val_acc_40_categories': best_val_acc,
    'knn_acc_unseen_categories': knn_acc
}

# ============================
# SAVE ALL RESULTS
# ============================

print("\n" + "="*50)
print("Saving All Day 2 Results")
print("="*50)

day2_results = {
    'experiment_1_fewshot': few_shot_results,
    'experiment_2_augmentation': augmentation_results,
    'experiment_3_transfer': transfer_results
}

save_results(day2_results, os.path.join(Config.RESULTS_DIR, 'day2_all_results.json'))

# Create summary report
print("\n" + "="*60)
print("DAY 2 SUMMARY REPORT")
print("="*60)

print("\n1. FEW-SHOT LEARNING:")
for i, n in enumerate(few_shot_results['samples']):
    print(f"   {n} samples/category: Test Acc = {few_shot_results['test_acc'][i]:.2f}%")

print("\n2. AUGMENTATION STRATEGIES:")
for i, strategy in enumerate(augmentation_results['strategy']):
    print(f"   {strategy}: Test Acc = {augmentation_results['test_acc'][i]:.2f}%")

print("\n3. TRANSFER LEARNING:")
print(f"   Validation Acc (40 categories): {best_val_acc:.2f}%")
print(f"   k-NN Acc (10 unseen categories): {knn_acc:.2f}%")

print("\n" + "="*60)
print("DAY 2 COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"\nAll results saved to: {Config.RESULTS_DIR}")

