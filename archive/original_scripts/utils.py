"""
Utility functions for QuickDraw sketch recognition project
Updated to work with .npy files
"""

import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_npy_file(filepath, max_samples=2000):
    """
    Load QuickDraw .npy file
    
    Args:
        filepath: Path to .npy file
        max_samples: Maximum number of samples to load
    
    Returns:
        numpy array of shape (N, 784) where each row is a flattened 28x28 image
    """
    data = np.load(filepath)
    
    # Limit to max_samples
    if len(data) > max_samples:
        # Use fixed seed for reproducibility
        np.random.seed(SEED)
        indices = np.random.choice(len(data), max_samples, replace=False)
        data = data[indices]
    
    return data

def npy_to_image(npy_data, size=64):
    """
    Convert numpy array (784,) to PIL Image
    
    Args:
        npy_data: Flattened 28x28 image (784,)
        size: Output size (default 64x64)
    
    Returns:
        PIL Image
    """
    # Reshape to 28x28
    img_array = npy_data.reshape(28, 28)
    
    # Convert to PIL Image (grayscale)
    img = Image.fromarray(img_array.astype('uint8'), mode='L')
    
    # Resize to desired size
    img = img.resize((size, size), Image.LANCZOS)
    
    return img

def prepare_dataset_from_npy(data_dir, categories, samples_per_category=2000, output_dir='processed_data', img_size=64):
    """
    Prepare dataset from .npy files
    
    Args:
        data_dir: Directory containing .npy files
        categories: List of category names
        samples_per_category: Number of samples to extract per category
        output_dir: Where to save processed images
        img_size: Image size (default 64x64)
    
    Returns:
        DataFrame with image paths and labels
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}\n"
            f"Run download_dataset.py first, or update DATA_DIR in local_config.py."
        )

    os.makedirs(output_dir, exist_ok=True)

    data_records = []

    print(f"\nProcessing {len(categories)} categories...")
    
    for category in categories:
        print(f"Processing {category}...", end=' ')
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        # Try different file naming conventions for .npy files
        npy_file = None
        possible_names = [
            os.path.join(data_dir, f"{category}.npy"),
            os.path.join(data_dir, f"{category.replace(' ', '_')}.npy"),
            os.path.join(data_dir, f"{category.replace(' ', '%20')}.npy"),
            os.path.join(data_dir, f"full_numpy_bitmap_{category}.npy"),
            os.path.join(data_dir, f"full_numpy_bitmap_{category.replace(' ', '_')}.npy"),
        ]
        
        for fname in possible_names:
            if os.path.exists(fname):
                npy_file = fname
                break
        
        if npy_file is None:
            print(f"⚠️  Warning: Could not find file for {category}")
            # List similar files for debugging
            available_files = [f for f in os.listdir(data_dir) if category.replace(' ', '_') in f or category.replace(' ', '%20') in f]
            if available_files:
                print(f"   Found similar files: {available_files}")
            continue
        
        try:
            # Load numpy data
            npy_data = load_npy_file(npy_file, max_samples=samples_per_category)
            print(f"loaded {len(npy_data)} samples", end=' ')
            
            # Convert to images and save
            for idx, img_data in enumerate(npy_data):
                try:
                    img = npy_to_image(img_data, size=img_size)
                    img_path = os.path.join(category_dir, f"{idx:04d}.png")
                    img.save(img_path)
                    
                    data_records.append({
                        'image_path': img_path,
                        'category': category,
                        'label': categories.index(category)
                    })
                except Exception as e:
                    print(f"Error processing {category} sample {idx}: {e}")
                    continue
            
            print("✓")
            
        except Exception as e:
            print(f"⚠️  Error loading {category}: {e}")
            continue
    
    df = pd.DataFrame(data_records)
    df.to_csv(os.path.join(output_dir, 'dataset.csv'), index=False)
    print(f"\n✓ Dataset prepared: {len(df)} images across {df['category'].nunique()} categories")
    
    return df

class QuickDrawDataset(Dataset):
    """PyTorch Dataset for QuickDraw sketches"""
    
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        label = self.df.iloc[idx]['label']
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(img_size=64, augment=True):
    """Get data transforms for training and validation"""
    
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def split_dataset(df, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Split dataset into train, validation, and test sets"""
    
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    for category in df['category'].unique():
        category_df = df[df['category'] == category].sample(frac=1, random_state=seed).reset_index(drop=True)
        
        n = len(category_df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_dfs.append(category_df[:train_end])
        val_dfs.append(category_df[train_end:val_end])
        test_dfs.append(category_df[val_end:])
    
    train_df = pd.concat(train_dfs, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_dfs, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_dfs, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return train_df, val_df, test_df

def plot_samples(df, categories, n_samples=5, save_path='sample_images.png'):
    """Plot sample images from each category"""
    
    n_categories = min(10, len(categories))
    fig, axes = plt.subplots(n_categories, n_samples, figsize=(15, 3*n_categories))
    
    if n_categories == 1:
        axes = axes.reshape(1, -1)
    
    for i, category in enumerate(categories[:n_categories]):
        category_df = df[df['category'] == category]
        
        if len(category_df) < n_samples:
            print(f"Warning: {category} has only {len(category_df)} samples")
            n_samples_actual = len(category_df)
        else:
            n_samples_actual = n_samples
            
        category_df = category_df.sample(n=n_samples_actual, random_state=42)
        
        for j, (_, row) in enumerate(category_df.iterrows()):
            img = Image.open(row['image_path'])
            axes[i, j].imshow(img, cmap='gray')
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_title(category, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Sample images saved to {save_path}")

def plot_class_distribution(df, save_path='class_distribution.png'):
    """Plot class distribution"""
    
    plt.figure(figsize=(15, 6))
    counts = df['category'].value_counts().sort_index()
    plt.bar(range(len(counts)), counts.values)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.title('Class Distribution', fontsize=14)
    plt.xticks(range(len(counts)), counts.index, rotation=90, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Class distribution plot saved to {save_path}")

def plot_confusion_matrix(y_true, y_pred, categories, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to {save_path}")

def save_results(results, filename='results.json'):
    """Save results to JSON file"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"✓ Results saved to {filename}")

def plot_training_curves(history, save_path='training_curves.png'):
    """Plot training and validation curves"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    ax2.plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Training curves saved to {save_path}")