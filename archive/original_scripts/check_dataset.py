"""
Dataset Verification Script
Run this FIRST to check your QuickDraw dataset structure and files
"""

import os
import numpy as np
from pathlib import Path
from local_config import DATA_DIR, DATA_ROOT

print("="*60)
print("QuickDraw Dataset Verification")
print("="*60)

# Check if directory exists
if not os.path.exists(DATA_DIR):
    print(f"\n❌ ERROR: Directory not found: {DATA_DIR}")
    print("\n📋 Available directories under DATA_ROOT:")
    if os.path.exists(DATA_ROOT):
        subdirs = os.listdir(DATA_ROOT)
        for subdir in subdirs:
            print(f"   - {subdir}")
        print(f"\n💡 Run download_dataset.py first, or update DATA_DIR in local_config.py")
    else:
        print(f"   {DATA_ROOT} does not exist yet. Run download_dataset.py first.")
    exit()

print(f"\n✓ Directory found: {DATA_DIR}")

# List all files
all_files = os.listdir(DATA_DIR)
print(f"\n📊 Total files in directory: {len(all_files)}")

# Filter by extension
npy_files = [f for f in all_files if f.endswith('.npy')]
ndjson_files = [f for f in all_files if f.endswith('.ndjson')]
other_files = [f for f in all_files if not (f.endswith('.npy') or f.endswith('.ndjson'))]

print(f"\n📁 File breakdown:")
print(f"   .npy files: {len(npy_files)}")
print(f"   .ndjson files: {len(ndjson_files)}")
print(f"   Other files: {len(other_files)}")

# Show sample files
if npy_files:
    print(f"\n✓ Found .npy files!")
    print(f"\n📋 Sample .npy files (first 10):")
    for f in npy_files[:10]:
        print(f"   - {f}")
    
    # Check file naming pattern
    print(f"\n🔍 Analyzing file naming patterns...")
    
    # Try to extract category names
    categories = []
    for f in npy_files:
        # Remove .npy extension
        name = f.replace('.npy', '')
        
        # Remove common prefixes
        name = name.replace('full_numpy_bitmap_', '')
        name = name.replace('full_simplified_', '')
        
        categories.append(name)
    
    print(f"\n📝 Detected categories (first 10):")
    for cat in categories[:10]:
        print(f"   - {cat}")
    
    # Check one file to see its structure
    print(f"\n🔬 Analyzing first .npy file structure...")
    first_file = os.path.join(DATA_DIR, npy_files[0])
    try:
        data = np.load(first_file)
        print(f"   File: {npy_files[0]}")
        print(f"   Shape: {data.shape}")
        print(f"   Dtype: {data.dtype}")
        print(f"   Min value: {data.min()}")
        print(f"   Max value: {data.max()}")
        
        if len(data.shape) == 2:
            print(f"\n   ✓ Format looks correct!")
            print(f"   ✓ Each row is a flattened 28x28 image (784 values)")
            print(f"   ✓ Total samples in this category: {data.shape[0]}")
        else:
            print(f"\n   ⚠️  Unexpected shape! Expected 2D array (N, 784)")
    except Exception as e:
        print(f"   ❌ Error loading file: {e}")
    
    print(f"\n" + "="*60)
    print("RECOMMENDED CATEGORIES LIST FOR YOUR CODE:")
    print("="*60)
    print("\nCATEGORIES = [")
    for i, cat in enumerate(categories[:50]):  # Show first 50
        print(f"    '{cat}',")
    print("]")
    
    print(f"\n✓ You have {len(npy_files)} categories available")
    print(f"✓ You can use up to 50 categories for your research")

elif ndjson_files:
    print(f"\n⚠️  Found .ndjson files instead of .npy files")
    print(f"\n📋 Sample .ndjson files (first 10):")
    for f in ndjson_files[:10]:
        print(f"   - {f}")
    print(f"\n💡 The code has been updated to work with .npy files.")
    print(f"   If you want to use .ndjson files, please convert them first.")
    
else:
    print(f"\n❌ No .npy or .ndjson files found!")
    print(f"\n📋 Files in directory:")
    for f in all_files[:20]:
        print(f"   - {f}")
    print(f"\n💡 Please check if your dataset is in the correct format.")

print(f"\n" + "="*60)
print("Next Steps:")
print("="*60)
print("1. Copy the CATEGORIES list printed above")
print("2. Update CATEGORIES in day1_setup_baseline.py")
print("3. Update DATA_DIR in day1_setup_baseline.py")
print("4. Run: python day1_setup_baseline.py")
print("="*60)