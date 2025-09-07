"""
Download dataset using Kaggle Hub
"""

import kagglehub
import os
import shutil
from pathlib import Path

def download_dataset():
    """Download the chest X-ray pneumonia dataset"""
    print("🏥 Downloading Chest X-Ray Pneumonia Dataset...")
    print("="*50)
    
    try:
        # Download latest version
        path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        print(f"✅ Dataset downloaded to: {path}")
        
        # Check if the dataset has the expected structure
        dataset_path = Path(path)
        print(f"\n📁 Dataset contents:")
        for item in dataset_path.iterdir():
            if item.is_dir():
                print(f"  📂 {item.name}/")
                for subitem in item.iterdir():
                    if subitem.is_dir():
                        print(f"    📂 {subitem.name}/")
                        # Count images
                        images = list(subitem.glob("*.jpg")) + list(subitem.glob("*.png"))
                        print(f"      📊 {len(images)} images")
        
        # Move to our data directory
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Copy the chest_xray folder to our data directory
        source_path = dataset_path / "chest_xray"
        target_path = data_dir / "chest_xray"
        
        if source_path.exists():
            if target_path.exists():
                print(f"\n🔄 Removing existing data directory...")
                shutil.rmtree(target_path)
            
            print(f"\n📋 Copying dataset to {target_path}...")
            shutil.copytree(source_path, target_path)
            print(f"✅ Dataset copied successfully!")
            
            # Verify the structure
            print(f"\n🔍 Verifying dataset structure...")
            expected_dirs = [
                "train/NORMAL",
                "train/PNEUMONIA", 
                "test/NORMAL",
                "test/PNEUMONIA"
            ]
            
            total_images = 0
            for dir_path in expected_dirs:
                full_path = target_path / dir_path
                if full_path.exists():
                    images = list(full_path.glob("*.jpg")) + list(full_path.glob("*.png"))
                    total_images += len(images)
                    print(f"  ✅ {dir_path}: {len(images)} images")
                else:
                    print(f"  ❌ {dir_path}: Not found")
            
            print(f"\n🎉 Dataset setup complete!")
            print(f"📊 Total images: {total_images}")
            print(f"📁 Dataset location: {target_path}")
            
            return True
            
        else:
            print(f"❌ Expected 'chest_xray' folder not found in {dataset_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading dataset: {str(e)}")
        return False

if __name__ == "__main__":
    success = download_dataset()
    if success:
        print("\n🚀 Ready to train! Run: python train.py")
    else:
        print("\n❌ Dataset download failed. Please check the error above.")
