"""
Data Download Script for Medical ViT
Downloads and organizes the chest X-ray dataset
"""

import os
import requests
import zipfile
from pathlib import Path
import shutil


def download_kaggle_dataset():
    """
    Instructions for downloading the Kaggle dataset
    """
    print("="*60)
    print("📥 DATASET DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print()
    print("1. Go to: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    print("2. Click 'Download' button")
    print("3. Extract the zip file")
    print("4. Move the 'chest_xray' folder to the 'data' directory")
    print()
    print("Expected structure:")
    print("data/")
    print("  chest_xray/")
    print("    train/")
    print("      NORMAL/")
    print("      PNEUMONIA/")
    print("    test/")
    print("      NORMAL/")
    print("      PNEUMONIA/")
    print()
    print("="*60)


def create_sample_data():
    """
    Create a small sample dataset for testing
    """
    print("Creating sample data structure...")
    
    # Create directories
    data_dir = Path("data/chest_xray")
    train_normal = data_dir / "train" / "NORMAL"
    train_pneumonia = data_dir / "train" / "PNEUMONIA"
    test_normal = data_dir / "test" / "NORMAL"
    test_pneumonia = data_dir / "test" / "PNEUMONIA"
    
    # Create all directories
    for dir_path in [train_normal, train_pneumonia, test_normal, test_pneumonia]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("✅ Sample data structure created!")
    print("📁 Please add your chest X-ray images to the appropriate folders")
    print()
    print("You can now run: python train.py")


def check_data_structure():
    """
    Check if the data is properly organized
    """
    data_dir = Path("data/chest_xray")
    
    if not data_dir.exists():
        print("❌ Data directory not found!")
        return False
    
    required_dirs = [
        "train/NORMAL",
        "train/PNEUMONIA", 
        "test/NORMAL",
        "test/PNEUMONIA"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not (data_dir / dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print("❌ Missing directories:")
        for dir_path in missing_dirs:
            print(f"   - {dir_path}")
        return False
    
    # Count images
    total_images = 0
    for dir_path in required_dirs:
        images = list((data_dir / dir_path).glob("*.jpg")) + list((data_dir / dir_path).glob("*.png"))
        total_images += len(images)
        print(f"✅ {dir_path}: {len(images)} images")
    
    print(f"✅ Total images: {total_images}")
    
    if total_images == 0:
        print("❌ No images found! Please add chest X-ray images to the directories.")
        return False
    
    return True


def main():
    print("🏥 Medical ViT - Data Setup")
    print("="*40)
    
    # Check if data already exists
    if check_data_structure():
        print("✅ Data structure is correct!")
        print("🚀 You can now run: python train.py")
        return
    
    print()
    print("Choose an option:")
    print("1. Download instructions for Kaggle dataset")
    print("2. Create sample data structure")
    print("3. Check data structure")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        download_kaggle_dataset()
    elif choice == "2":
        create_sample_data()
    elif choice == "3":
        check_data_structure()
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
