"""
Day 4: Alternative Models Testing and Comparison
QuickDraw Sketch Recognition Project

This script tests multiple machine learning models and compares them with ResNet18 baseline:
1. Traditional ML: Random Forest, SVM, k-NN
2. Lightweight CNNs: MobileNetV2, EfficientNet-B0
3. Medium CNNs: ResNet34, DenseNet121
4. Modern Architectures: Vision Transformer (ViT)

All models evaluated on same test set for fair comparison.
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
from torchvision import models, transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.decomposition import PCA
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
    RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results/day4')
    MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')

    # Dataset parameters
    IMG_SIZE = 64
    NUM_CLASSES = 50

    # Training parameters
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 15  # Reduced for faster comparison
    DEVICE = get_device()
    NUM_WORKERS = 2
    PIN_MEMORY = DEVICE == 'cuda'

    # Traditional ML parameters
    PCA_COMPONENTS = 100  # For dimensionality reduction

    # Seed
    SEED = 42

# Create directories
os.makedirs(Config.RESULTS_DIR, exist_ok=True)

# Set seed
set_seed(Config.SEED)

print("="*80)
print("DAY 4: ALTERNATIVE MODELS TESTING AND COMPARISON")
print("="*80)
print(f"\nUsing device: {Config.DEVICE}")

# ============================
# LOAD PROCESSED DATA
# ============================

print("\n" + "="*80)
print("STEP 1: Loading Processed Data")
print("="*80)

# Check if processed data exists
train_csv = os.path.join(Config.PROCESSED_DIR, 'train.csv')
val_csv = os.path.join(Config.PROCESSED_DIR, 'val.csv')
test_csv = os.path.join(Config.PROCESSED_DIR, 'test.csv')

if not os.path.exists(train_csv):
    print("❌ ERROR: Processed data not found!")
    print(f"   Looking for: {train_csv}")
    print("\n📋 Available directories:")
    if os.path.exists(Config.OUTPUT_DIR):
        subdirs = [d for d in os.listdir(Config.OUTPUT_DIR) if os.path.isdir(os.path.join(Config.OUTPUT_DIR, d))]
        print(f"   {subdirs}")
    
    print("\n💡 SOLUTION:")
    print("   Day 4 requires Day 1 to be completed first in this notebook.")
    print("   Please run Day 1 setup script first to generate processed data:")
    print("   !python day1_setup_baseline.py")
    print("\n   OR if you ran Day 1 in a different notebook:")
    print("   1. Download processed_data/ folder from that notebook")
    print("   2. Upload it to this notebook")
    exit(1)

print("✓ Found processed data files")
train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)
test_df = pd.read_csv(test_csv)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ============================
# PREPARE DATALOADERS
# ============================

print("\n" + "="*80)
print("STEP 2: Preparing DataLoaders")
print("="*80)

train_transform = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
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

print("DataLoaders ready!")

# ============================
# MODEL DEFINITIONS
# ============================

def create_model(model_name, num_classes=50, pretrained=True):
    """Create model based on name"""
    
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        
    elif model_name == 'simple_cnn':
        # Simple custom CNN for baseline
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

# ============================
# TRAINING FUNCTIONS
# ============================

def train_model(model, train_loader, val_loader, model_name, num_epochs=15):
    """Train a deep learning model"""
    
    print(f"\n{'='*80}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*80}")
    
    model = model.to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    best_val_acc = 0.0
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=False)
        for images, labels in pbar:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                             'acc': f'{100.*train_correct/train_total:.2f}%'})
        
        train_acc = 100. * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', leave=False)
            for images, labels in pbar:
                images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    training_time = time.time() - start_time
    
    return model, best_val_acc, history, training_time

def evaluate_model(model, test_loader):
    """Evaluate model on test set"""
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing', leave=False):
            images = images.to(Config.DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    acc = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average='weighted') * 100
    
    return acc, f1, all_preds, all_labels

# ============================
# TRADITIONAL ML FUNCTIONS
# ============================

def extract_features(loader):
    """Extract flattened features from images"""
    
    features = []
    labels = []
    
    for images, lbls in tqdm(loader, desc='Extracting features', leave=False):
        # Convert to grayscale and flatten
        batch_features = []
        for img in images:
            # Average across channels to get grayscale
            gray = img.mean(dim=0).numpy().flatten()
            batch_features.append(gray)
        
        features.extend(batch_features)
        labels.extend(lbls.numpy())
    
    return np.array(features), np.array(labels)

def train_traditional_ml(model, X_train, y_train, X_val, y_val, model_name):
    """Train traditional ML model"""
    
    print(f"\n{'='*80}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # Apply PCA for dimensionality reduction
    print(f"Applying PCA (reducing to {Config.PCA_COMPONENTS} components)...")
    pca = PCA(n_components=Config.PCA_COMPONENTS, random_state=Config.SEED)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Train model
    print(f"Training {model_name}...")
    model.fit(X_train_pca, y_train)
    
    training_time = time.time() - start_time
    
    # Validation accuracy
    val_pred = model.predict(X_val_pca)
    val_acc = accuracy_score(y_val, val_pred) * 100
    
    print(f"Training completed in {training_time/60:.2f} minutes")
    print(f"Validation Accuracy: {val_acc:.2f}%")
    
    return model, pca, val_acc, training_time

def evaluate_traditional_ml(model, pca, X_test, y_test):
    """Evaluate traditional ML model"""
    
    X_test_pca = pca.transform(X_test)
    y_pred = model.predict(X_test_pca)
    
    acc = accuracy_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred, average='weighted') * 100
    
    return acc, f1, y_pred

# ============================
# MAIN EXPERIMENT
# ============================

print("\n" + "="*80)
print("STEP 3: Running Model Comparison Experiments")
print("="*80)

results = []

# ============================
# EXPERIMENT 1: DEEP LEARNING MODELS
# ============================

print("\n" + "="*80)
print("EXPERIMENT 1: DEEP LEARNING MODELS")
print("="*80)

dl_models = {
    'Simple CNN': 'simple_cnn',
    'MobileNetV2': 'mobilenet_v2',
    'ResNet18': 'resnet18',
    'ResNet34': 'resnet34',
    'DenseNet121': 'densenet121',
    'EfficientNet-B0': 'efficientnet_b0'
}

for model_display_name, model_name in dl_models.items():
    try:
        print(f"\n{'*'*80}")
        print(f"Testing: {model_display_name}")
        print(f"{'*'*80}")
        
        # Create model
        model = create_model(model_name, num_classes=Config.NUM_CLASSES, pretrained=True)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Train
        model, best_val_acc, history, training_time = train_model(
            model, train_loader, val_loader, model_display_name, num_epochs=Config.NUM_EPOCHS
        )
        
        # Evaluate
        test_acc, test_f1, test_preds, test_labels = evaluate_model(model, test_loader)
        
        print(f"\n{model_display_name} Results:")
        print(f"  Best Val Acc: {best_val_acc:.2f}%")
        print(f"  Test Acc: {test_acc:.2f}%")
        print(f"  Test F1: {test_f1:.2f}%")
        print(f"  Training Time: {training_time/60:.2f} minutes")
        
        # Store results
        results.append({
            'Model': model_display_name,
            'Type': 'Deep Learning',
            'Parameters': f"{total_params/1e6:.2f}M",
            'Val Accuracy': f"{best_val_acc:.2f}%",
            'Test Accuracy': f"{test_acc:.2f}%",
            'Test F1-Score': f"{test_f1:.2f}%",
            'Training Time (min)': f"{training_time/60:.2f}"
        })
        
        # Save model
        torch.save(model.state_dict(), 
                  os.path.join(Config.MODELS_DIR, f'{model_name}_best.pth'))
        
    except Exception as e:
        print(f"❌ Error with {model_display_name}: {e}")
        results.append({
            'Model': model_display_name,
            'Type': 'Deep Learning',
            'Parameters': 'N/A',
            'Val Accuracy': 'Error',
            'Test Accuracy': 'Error',
            'Test F1-Score': 'Error',
            'Training Time (min)': 'N/A'
        })

# ============================
# EXPERIMENT 2: TRADITIONAL ML MODELS
# ============================

print("\n" + "="*80)
print("EXPERIMENT 2: TRADITIONAL MACHINE LEARNING MODELS")
print("="*80)

print("\nExtracting features from images...")
X_train, y_train = extract_features(train_loader)
X_val, y_val = extract_features(val_loader)
X_test, y_test = extract_features(test_loader)

print(f"Training features shape: {X_train.shape}")
print(f"Validation features shape: {X_val.shape}")
print(f"Test features shape: {X_test.shape}")

traditional_models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20, 
                                           random_state=Config.SEED, n_jobs=-1),
    'k-NN (k=5)': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'SVM (Linear)': SVC(kernel='linear', random_state=Config.SEED, max_iter=1000)
}

for model_name, model in traditional_models.items():
    try:
        print(f"\n{'*'*80}")
        print(f"Testing: {model_name}")
        print(f"{'*'*80}")
        
        # Train
        trained_model, pca, val_acc, training_time = train_traditional_ml(
            model, X_train, y_train, X_val, y_val, model_name
        )
        
        # Evaluate
        test_acc, test_f1, test_preds = evaluate_traditional_ml(
            trained_model, pca, X_test, y_test
        )
        
        print(f"\n{model_name} Results:")
        print(f"  Val Acc: {val_acc:.2f}%")
        print(f"  Test Acc: {test_acc:.2f}%")
        print(f"  Test F1: {test_f1:.2f}%")
        print(f"  Training Time: {training_time/60:.2f} minutes")
        
        # Store results
        results.append({
            'Model': model_name,
            'Type': 'Traditional ML',
            'Parameters': f"PCA: {Config.PCA_COMPONENTS}",
            'Val Accuracy': f"{val_acc:.2f}%",
            'Test Accuracy': f"{test_acc:.2f}%",
            'Test F1-Score': f"{test_f1:.2f}%",
            'Training Time (min)': f"{training_time/60:.2f}"
        })
        
    except Exception as e:
        print(f"❌ Error with {model_name}: {e}")
        results.append({
            'Model': model_name,
            'Type': 'Traditional ML',
            'Parameters': 'N/A',
            'Val Accuracy': 'Error',
            'Test Accuracy': 'Error',
            'Test F1-Score': 'Error',
            'Training Time (min)': 'N/A'
        })

# ============================
# STEP 4: RESULTS ANALYSIS
# ============================

print("\n" + "="*80)
print("STEP 4: Analyzing and Displaying Results")
print("="*80)

# Create results DataFrame
results_df = pd.DataFrame(results)

# Save results
results_df.to_csv(os.path.join(Config.RESULTS_DIR, 'model_comparison.csv'), index=False)
print("\n✓ Results saved to model_comparison.csv")

# Print comparison table
print("\n" + "="*80)
print("MODEL COMPARISON TABLE")
print("="*80)
print(results_df.to_string(index=False))

# ============================
# STEP 5: VISUALIZATION (DISPLAY INLINE)
# ============================

print("\n" + "="*80)
print("STEP 5: Creating and Displaying Visualizations")
print("="*80)

# Extract numeric values for plotting
try:
    results_df['Test Acc (numeric)'] = results_df['Test Accuracy'].str.rstrip('%').astype(float)
    results_df['Test F1 (numeric)'] = results_df['Test F1-Score'].str.rstrip('%').astype(float)
    results_df['Training Time (numeric)'] = results_df['Training Time (min)'].astype(float)
except:
    print("⚠️  Some models may have failed. Filtering errors...")
    results_df = results_df[results_df['Test Accuracy'] != 'Error']
    results_df['Test Acc (numeric)'] = results_df['Test Accuracy'].str.rstrip('%').astype(float)
    results_df['Test F1 (numeric)'] = results_df['Test F1-Score'].str.rstrip('%').astype(float)
    results_df['Training Time (numeric)'] = results_df['Training Time (min)'].astype(float)

# Sort by test accuracy
results_df = results_df.sort_values('Test Acc (numeric)', ascending=False)

# Plot 1: Model Comparison - Accuracy and F1-Score
print("\n📊 Visualization 1: Model Performance Comparison")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Accuracy comparison
colors = ['#2ecc71' if t == 'Deep Learning' else '#e74c3c' for t in results_df['Type']]
bars1 = ax1.barh(results_df['Model'], results_df['Test Acc (numeric)'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

for i, (bar, acc) in enumerate(zip(bars1, results_df['Test Acc (numeric)'])):
    ax1.text(acc + 1, bar.get_y() + bar.get_height()/2, f'{acc:.2f}%',
            va='center', fontsize=10, fontweight='bold')

ax1.set_xlabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
ax1.set_title('Model Comparison: Test Accuracy', fontsize=15, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
ax1.set_xlim(0, max(results_df['Test Acc (numeric)']) + 10)

# F1-Score comparison
bars2 = ax2.barh(results_df['Model'], results_df['Test F1 (numeric)'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

for i, (bar, f1) in enumerate(zip(bars2, results_df['Test F1 (numeric)'])):
    ax2.text(f1 + 1, bar.get_y() + bar.get_height()/2, f'{f1:.2f}%',
            va='center', fontsize=10, fontweight='bold')

ax2.set_xlabel('Test F1-Score (%)', fontsize=13, fontweight='bold')
ax2.set_title('Model Comparison: Test F1-Score', fontsize=15, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
ax2.set_xlim(0, max(results_df['Test F1 (numeric)']) + 10)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2ecc71', label='Deep Learning'),
                  Patch(facecolor='#e74c3c', label='Traditional ML')]
ax1.legend(handles=legend_elements, loc='lower right', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(Config.RESULTS_DIR, 'model_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()  # Display inline
print("✓ Saved: model_comparison.png")

# Plot 2: Scatter plot - Accuracy vs Training Time
print("\n📊 Visualization 2: Efficiency Analysis")
fig, ax = plt.subplots(figsize=(12, 8))

for model_type in ['Deep Learning', 'Traditional ML']:
    df_type = results_df[results_df['Type'] == model_type]
    color = '#2ecc71' if model_type == 'Deep Learning' else '#e74c3c'
    ax.scatter(df_type['Training Time (numeric)'], df_type['Test Acc (numeric)'],
              s=200, alpha=0.7, color=color, edgecolor='black', linewidth=2, label=model_type)
    
    for _, row in df_type.iterrows():
        ax.annotate(row['Model'], 
                   (row['Training Time (numeric)'], row['Test Acc (numeric)']),
                   xytext=(10, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))

ax.set_xlabel('Training Time (minutes)', fontsize=13, fontweight='bold')
ax.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Model Efficiency: Accuracy vs Training Time', fontsize=15, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(Config.RESULTS_DIR, 'efficiency_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()  # Display inline
print("✓ Saved: efficiency_comparison.png")

# Plot 3: Summary Statistics
print("\n📊 Visualization 3: Performance Summary by Model Type")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Group by type
type_summary = results_df.groupby('Type').agg({
    'Test Acc (numeric)': ['mean', 'max', 'min'],
    'Training Time (numeric)': ['mean', 'max', 'min']
}).round(2)

# Accuracy summary
model_types = results_df['Type'].unique()
x_pos = np.arange(len(model_types))

for i, model_type in enumerate(model_types):
    type_data = results_df[results_df['Type'] == model_type]
    color = '#2ecc71' if model_type == 'Deep Learning' else '#e74c3c'
    
    # Box plot data
    bp = ax1.boxplot([type_data['Test Acc (numeric)'].values], 
                     positions=[i], widths=0.6,
                     patch_artist=True,
                     boxprops=dict(facecolor=color, alpha=0.7),
                     medianprops=dict(color='black', linewidth=2),
                     whiskerprops=dict(color='black', linewidth=1.5),
                     capprops=dict(color='black', linewidth=1.5))

ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Accuracy Distribution by Model Type', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(model_types)
ax1.grid(axis='y', alpha=0.3)

# Training time summary
for i, model_type in enumerate(model_types):
    type_data = results_df[results_df['Type'] == model_type]
    color = '#2ecc71' if model_type == 'Deep Learning' else '#e74c3c'
    
    bp = ax2.boxplot([type_data['Training Time (numeric)'].values], 
                     positions=[i], widths=0.6,
                     patch_artist=True,
                     boxprops=dict(facecolor=color, alpha=0.7),
                     medianprops=dict(color='black', linewidth=2),
                     whiskerprops=dict(color='black', linewidth=1.5),
                     capprops=dict(color='black', linewidth=1.5))

ax2.set_ylabel('Training Time (minutes)', fontsize=12, fontweight='bold')
ax2.set_title('Training Time Distribution by Model Type', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(model_types)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()  # Display inline
print("✓ Summary statistics displayed")

# ============================
# STEP 6: DETAILED ANALYSIS
# ============================

print("\n" + "="*80)
print("STEP 6: Detailed Statistical Analysis")
print("="*80)

# Best models
best_accuracy = results_df.iloc[0]
print(f"\n🏆 BEST OVERALL MODEL:")
print(f"   Model: {best_accuracy['Model']}")
print(f"   Type: {best_accuracy['Type']}")
print(f"   Test Accuracy: {best_accuracy['Test Accuracy']}")
print(f"   Test F1-Score: {best_accuracy['Test F1-Score']}")
print(f"   Training Time: {best_accuracy['Training Time (min)']} minutes")

# Best by category
dl_models = results_df[results_df['Type'] == 'Deep Learning']
ml_models = results_df[results_df['Type'] == 'Traditional ML']

if len(dl_models) > 0:
    best_dl = dl_models.iloc[0]
    print(f"\n🥇 BEST DEEP LEARNING MODEL:")
    print(f"   Model: {best_dl['Model']}")
    print(f"   Test Accuracy: {best_dl['Test Accuracy']}")

if len(ml_models) > 0:
    best_ml = ml_models.iloc[0]
    print(f"\n🥈 BEST TRADITIONAL ML MODEL:")
    print(f"   Model: {best_ml['Model']}")
    print(f"   Test Accuracy: {best_ml['Test Accuracy']}")

# Fastest model
fastest = results_df.loc[results_df['Training Time (numeric)'].idxmin()]
print(f"\n⚡ FASTEST MODEL:")
print(f"   Model: {fastest['Model']}")
print(f"   Training Time: {fastest['Training Time (min)']} minutes")
print(f"   Test Accuracy: {fastest['Test Accuracy']}")

# Most efficient (accuracy per minute)
results_df['Efficiency'] = results_df['Test Acc (numeric)'] / results_df['Training Time (numeric)']
most_efficient = results_df.loc[results_df['Efficiency'].idxmax()]
print(f"\n💎 MOST EFFICIENT MODEL (Accuracy/Time):")
print(f"   Model: {most_efficient['Model']}")
print(f"   Efficiency: {most_efficient['Efficiency']:.2f}% per minute")
print(f"   Test Accuracy: {most_efficient['Test Accuracy']}")
print(f"   Training Time: {most_efficient['Training Time (min)']} minutes")

# Performance gap analysis
if len(dl_models) > 0 and len(ml_models) > 0:
    avg_dl_acc = dl_models['Test Acc (numeric)'].mean()
    avg_ml_acc = ml_models['Test Acc (numeric)'].mean()
    gap = avg_dl_acc - avg_ml_acc
    
    print(f"\n📊 DEEP LEARNING vs TRADITIONAL ML:")
    print(f"   Average DL Accuracy: {avg_dl_acc:.2f}%")
    print(f"   Average ML Accuracy: {avg_ml_acc:.2f}%")
    print(f"   Performance Gap: {gap:.2f}%")
    print(f"   DL Advantage: {(gap/avg_ml_acc)*100:.1f}% relative improvement")

# ============================
# FINAL SUMMARY
# ============================

print("\n" + "="*80)
print("DAY 4 COMPLETED SUCCESSFULLY!")
print("="*80)

print("\n📊 Summary:")
print(f"   Models Tested: {len(results_df)}")
print(f"   Best Model: {results_df.iloc[0]['Model']}")
print(f"   Best Accuracy: {results_df.iloc[0]['Test Accuracy']}")
print(f"   Best F1-Score: {results_df.iloc[0]['Test F1-Score']}")

print(f"\n📁 Outputs saved to: {Config.RESULTS_DIR}")
print("   - model_comparison.csv")
print("   - model_comparison.png")
print("   - efficiency_comparison.png")

print("\n" + "="*80)