"""
Setup Script for Medical ViT Project
Installs dependencies and sets up the project
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False


def main():
    print("🏥 Medical ViT Project Setup")
    print("="*50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ is required!")
        print(f"Current version: {python_version.major}.{python_version.minor}")
        return 1
    
    print(f"✅ Python version: {python_version.major}.{python_version.minor}")
    
    # Install PyTorch first (with CUDA support if available)
    print("\n📦 Installing PyTorch...")
    if run_command("python -c 'import torch; print(torch.cuda.is_available())'", "Checking CUDA availability"):
        print("🚀 CUDA is available! Installing PyTorch with CUDA support...")
        torch_command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    else:
        print("💻 Installing PyTorch for CPU...")
        torch_command = "pip install torch torchvision torchaudio"
    
    if not run_command(torch_command, "Installing PyTorch"):
        return 1
    
    # Install other requirements
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        return 1
    
    # Create necessary directories
    print("\n📁 Creating project directories...")
    directories = [
        "data",
        "models", 
        "results",
        "logs",
        "notebooks"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Check if data exists
    data_path = Path("data/chest_xray")
    if not data_path.exists():
        print("\n📥 Data setup required!")
        print("Run: python download_data.py")
        print("This will provide instructions for downloading the dataset.")
    else:
        print("✅ Data directory found!")
    
    print("\n" + "="*50)
    print("🎉 Setup completed successfully!")
    print("="*50)
    print("\nNext steps:")
    print("1. Download dataset: python download_data.py")
    print("2. Train model: python train.py")
    print("3. Run demo: streamlit run app.py")
    print("\nHappy coding! 🚀")
    
    return 0


if __name__ == "__main__":
    exit(main())
