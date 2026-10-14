"""
Master Pipeline v2: Cross-Dataset Validation (QuickDraw -> TU-Berlin)

Extends the Day 1-5 QuickDraw pipeline with a cross-dataset generalization
test: train SketchNet on QuickDraw, then measure how well it transfers to
TU-Berlin images of the *same* categories via three approaches:
  1. Zero-shot   - evaluate the QuickDraw-trained model directly, no retraining
  2. Fine-tuned  - continue training that model on a little TU-Berlin data
  3. From-scratch - train a fresh model on TU-Berlin data only

Runs entirely locally (CUDA / Apple Silicon MPS / CPU auto-detected).

--------------------------------------------------------------------------
IMPORTANT SCOPE NOTE vs. the original plan
--------------------------------------------------------------------------
The original plan (written for a Kaggle GPU notebook) called for the full
TU-Berlin dataset (250 categories, ~20k images, several GB) and a 9-11 hour
real run. This machine doesn't have room for that full dataset alongside
everything else already downloaded. Instead:
  - We use only the categories confirmed to exist in BOTH datasets
    (local_config.TUBERLIN_SHARED_CATEGORIES), fetched file-by-file instead
    of as one large archive (see download_tuberlin_subset.py) to keep disk
    usage in the tens of MB instead of gigabytes.
  - Because the category *names* match between datasets, zero-shot transfer
    doesn't need a new classifier head -- the label space is identical, so
    we can evaluate the QuickDraw-trained model on TU-Berlin directly.
  - SMOKE_TEST=True (default) uses a handful of categories/samples/epochs to
    verify the whole mechanism end-to-end in a couple of minutes. Turn it
    off for a larger (still deliberately modest) real run; see Config below
    to scale it up further if you free up more disk first.
--------------------------------------------------------------------------
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from local_config import (
    DATA_DIR, OUTPUT_DIR, TUBERLIN_SHARED_CATEGORIES, get_device,
)
from utils import set_seed, prepare_dataset_from_npy, QuickDrawDataset, get_transforms, split_dataset
from sketchnet_model import SketchNet
from download_tuberlin_subset import ensure_tuberlin_categories

# ============================
# CONFIGURATION
# ============================

class Config:
    SMOKE_TEST = True  # <-- flip to False for the (still modest) real run

    OUTPUT_DIR = OUTPUT_DIR
    PROCESSED_DIR = os.path.join(OUTPUT_DIR, 'processed_data_v2')
    RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results/master_v2')
    MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')

    IMG_SIZE = 64
    DEVICE = get_device()
    NUM_WORKERS = 0  # datasets here are tiny; multiprocessing overhead isn't worth it
    PIN_MEMORY = DEVICE == 'cuda'
    SEED = 42

    if SMOKE_TEST:
        CATEGORIES = TUBERLIN_SHARED_CATEGORIES[:5]
        QUICKDRAW_SAMPLES_PER_CATEGORY = 40
        TUBERLIN_SAMPLES_PER_CATEGORY = 8
        BATCH_SIZE = 8
        EPOCHS_SOURCE = 1
        EPOCHS_FINETUNE = 1
        EPOCHS_SCRATCH = 1
        LEARNING_RATE = 0.001
        LEARNING_RATE_FINETUNE = 0.0005
    else:
        CATEGORIES = TUBERLIN_SHARED_CATEGORIES
        QUICKDRAW_SAMPLES_PER_CATEGORY = 800
        TUBERLIN_SAMPLES_PER_CATEGORY = 60
        BATCH_SIZE = 32
        EPOCHS_SOURCE = 12
        EPOCHS_FINETUNE = 8
        EPOCHS_SCRATCH = 12
        LEARNING_RATE = 0.001
        LEARNING_RATE_FINETUNE = 0.0001

    NUM_CLASSES = len(CATEGORIES)


os.makedirs(Config.RESULTS_DIR, exist_ok=True)
os.makedirs(Config.MODELS_DIR, exist_ok=True)
set_seed(Config.SEED)

print("=" * 80)
print("MASTER PIPELINE v2: CROSS-DATASET VALIDATION (QuickDraw -> TU-Berlin)")
print("=" * 80)
print(f"Mode: {'SMOKE TEST' if Config.SMOKE_TEST else 'REAL RUN'}")
print(f"Device: {Config.DEVICE}")
print(f"Categories ({Config.NUM_CLASSES}): {Config.CATEGORIES}")

category_to_idx = {cat: idx for idx, cat in enumerate(Config.CATEGORIES)}


# ============================
# STEP 1: QUICKDRAW SUBSET
# ============================

print("\n" + "=" * 80)
print("STEP 1: Preparing QuickDraw subset")
print("=" * 80)

qd_df = prepare_dataset_from_npy(
    data_dir=DATA_DIR,
    categories=Config.CATEGORIES,
    samples_per_category=Config.QUICKDRAW_SAMPLES_PER_CATEGORY,
    output_dir=Config.PROCESSED_DIR,
    img_size=Config.IMG_SIZE,
)
assert len(qd_df) > 0, "No QuickDraw images were prepared -- check DATA_DIR/categories."

qd_train_df, qd_val_df, _ = split_dataset(qd_df, train_ratio=0.7, val_ratio=0.15, seed=Config.SEED)
print(f"QuickDraw train/val: {len(qd_train_df)}/{len(qd_val_df)}")


# ============================
# STEP 2: TU-BERLIN SUBSET
# ============================

print("\n" + "=" * 80)
print("STEP 2: Preparing TU-Berlin subset")
print("=" * 80)

tuberlin_dir = ensure_tuberlin_categories(Config.CATEGORIES, Config.TUBERLIN_SAMPLES_PER_CATEGORY)

tb_records = []
for cat in Config.CATEGORIES:
    cat_dir = os.path.join(tuberlin_dir, cat)
    if not os.path.isdir(cat_dir):
        continue
    for fname in sorted(os.listdir(cat_dir)):
        tb_records.append({
            'image_path': os.path.join(cat_dir, fname),
            'category': cat,
            'label': category_to_idx[cat],
        })

tb_df = pd.DataFrame(tb_records)
assert len(tb_df) > 0, "No TU-Berlin images found -- check TUBERLIN_SHARED_CATEGORIES / network access."
print(f"TU-Berlin images: {len(tb_df)}")

tb_train_df, tb_val_df, tb_test_df = split_dataset(tb_df, train_ratio=0.6, val_ratio=0.2, seed=Config.SEED)
print(f"TU-Berlin train/val/test: {len(tb_train_df)}/{len(tb_val_df)}/{len(tb_test_df)}")


class TUBerlinDataset(torch.utils.data.Dataset):
    """Simple image-folder-style dataset for the TU-Berlin subset."""
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, row['label']


# ============================
# DATALOADERS
# ============================

train_transform, val_transform = get_transforms(img_size=Config.IMG_SIZE, augment=True)

def make_loader(dataset, shuffle):
    n = len(dataset)
    if n == 0:
        return None
    batch_size = min(Config.BATCH_SIZE, n)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                       num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)

qd_train_loader = make_loader(QuickDrawDataset(qd_train_df, transform=train_transform), True)
qd_val_loader = make_loader(QuickDrawDataset(qd_val_df, transform=val_transform), False)

tb_train_loader = make_loader(TUBerlinDataset(tb_train_df, transform=train_transform), True)
tb_val_loader = make_loader(TUBerlinDataset(tb_val_df, transform=val_transform), False)
tb_test_loader = make_loader(TUBerlinDataset(tb_test_df, transform=val_transform), False)


# ============================
# TRAIN / EVAL HELPERS
# ============================

def train_epochs(model, train_loader, val_loader, num_epochs, lr, device, desc="Training"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        correct, total = 0, 0
        pbar = tqdm(train_loader, desc=f'{desc} [{epoch+1}/{num_epochs}]', leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        train_acc = 100. * correct / max(total, 1)

        val_acc = evaluate(model, val_loader, device)[0] if val_loader is not None else float('nan')
        best_val_acc = max(best_val_acc, val_acc) if val_loader is not None else best_val_acc
        print(f"  {desc} epoch {epoch+1}/{num_epochs}: train_acc={train_acc:.2f}%, val_acc={val_acc:.2f}%")

    return model, best_val_acc


def evaluate(model, loader, device):
    if loader is None:
        return float('nan'), float('nan')
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy() if torch.is_tensor(labels) else labels)
    acc = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
    return acc, f1


# ============================
# STEP 3: TRAIN SOURCE MODEL ON QUICKDRAW
# ============================

print("\n" + "=" * 80)
print("STEP 3: Training source model on QuickDraw")
print("=" * 80)

source_model = SketchNet(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
total_params = sum(p.numel() for p in source_model.parameters())
print(f"SketchNet parameters: {total_params:,}")

start = time.time()
source_model, source_val_acc = train_epochs(
    source_model, qd_train_loader, qd_val_loader,
    Config.EPOCHS_SOURCE, Config.LEARNING_RATE, Config.DEVICE, desc="QuickDraw"
)
source_time = time.time() - start
print(f"\n✓ Source model trained in {source_time/60:.2f} min. Best QuickDraw val acc: {source_val_acc:.2f}%")

torch.save({'model_state_dict': source_model.state_dict(), 'categories': Config.CATEGORIES},
           os.path.join(Config.MODELS_DIR, 'master_v2_source_quickdraw.pth'))


# ============================
# EXPERIMENT 1: ZERO-SHOT TRANSFER
# ============================

print("\n" + "=" * 80)
print("EXPERIMENT 1: Zero-shot transfer (QuickDraw -> TU-Berlin)")
print("=" * 80)

zeroshot_acc, zeroshot_f1 = evaluate(source_model, tb_test_loader, Config.DEVICE)
print(f"Zero-shot: acc={zeroshot_acc:.2f}%, f1={zeroshot_f1:.2f}%")


# ============================
# EXPERIMENT 2: FINE-TUNING
# ============================

print("\n" + "=" * 80)
print("EXPERIMENT 2: Fine-tuning on TU-Berlin")
print("=" * 80)

finetune_model = SketchNet(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
finetune_model.load_state_dict(source_model.state_dict())

start = time.time()
finetune_model, _ = train_epochs(
    finetune_model, tb_train_loader, tb_val_loader,
    Config.EPOCHS_FINETUNE, Config.LEARNING_RATE_FINETUNE, Config.DEVICE, desc="Fine-tune"
)
finetune_time = time.time() - start

finetune_acc, finetune_f1 = evaluate(finetune_model, tb_test_loader, Config.DEVICE)
print(f"Fine-tuned: acc={finetune_acc:.2f}%, f1={finetune_f1:.2f}%, time={finetune_time/60:.2f} min")


# ============================
# EXPERIMENT 3: FROM SCRATCH
# ============================

print("\n" + "=" * 80)
print("EXPERIMENT 3: From scratch on TU-Berlin")
print("=" * 80)

scratch_model = SketchNet(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)

start = time.time()
scratch_model, _ = train_epochs(
    scratch_model, tb_train_loader, tb_val_loader,
    Config.EPOCHS_SCRATCH, Config.LEARNING_RATE, Config.DEVICE, desc="From-scratch"
)
scratch_time = time.time() - start

scratch_acc, scratch_f1 = evaluate(scratch_model, tb_test_loader, Config.DEVICE)
print(f"From-scratch: acc={scratch_acc:.2f}%, f1={scratch_f1:.2f}%, time={scratch_time/60:.2f} min")


# ============================
# RESULTS
# ============================

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

results_df = pd.DataFrame({
    'Approach': ['Zero-Shot', 'Fine-Tuning', 'From Scratch'],
    'Accuracy (%)': [zeroshot_acc, finetune_acc, scratch_acc],
    'F1-Score (%)': [zeroshot_f1, finetune_f1, scratch_f1],
    'Time (min)': [0, finetune_time / 60, scratch_time / 60],
})
print("\n" + results_df.to_string(index=False))
results_df.to_csv(os.path.join(Config.RESULTS_DIR, 'cross_dataset_results.csv'), index=False)

results_json = {
    'smoke_test': Config.SMOKE_TEST,
    'categories': Config.CATEGORIES,
    'quickdraw_source_val_acc': source_val_acc,
    'zero_shot': {'accuracy': zeroshot_acc, 'f1_score': zeroshot_f1},
    'fine_tuned': {'accuracy': finetune_acc, 'f1_score': finetune_f1, 'time_minutes': finetune_time / 60},
    'from_scratch': {'accuracy': scratch_acc, 'f1_score': scratch_f1, 'time_minutes': scratch_time / 60},
}
with open(os.path.join(Config.RESULTS_DIR, 'cross_dataset_results.json'), 'w') as f:
    json.dump(results_json, f, indent=2)

if not Config.SMOKE_TEST:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#e74c3c', '#f39c12', '#2ecc71']
    bars = ax.bar(results_df['Approach'], results_df['Accuracy (%)'], color=colors, alpha=0.85, edgecolor='black')
    for bar, acc in zip(bars, results_df['Accuracy (%)']):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1, f'{acc:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('QuickDraw -> TU-Berlin Cross-Dataset Validation')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, 'cross_dataset_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved plot to {Config.RESULTS_DIR}/cross_dataset_comparison.png")

print("\n" + "=" * 80)
if Config.SMOKE_TEST:
    print("SMOKE TEST PASSED. Set Config.SMOKE_TEST = False and rerun for real.")
else:
    print("MASTER PIPELINE v2 COMPLETE.")
print("=" * 80)
