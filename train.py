"""
Main Training Script for Medical ViT
Run this script to train the model from scratch
"""

import os
import sys
import yaml
import torch
import argparse
from datetime import datetime

# Add src to path
sys.path.append('src')

from src.trainer import train_model
from src.data_loader import DataManager
from src.model import create_model
from src.evaluator import evaluate_trained_model


def main():
    parser = argparse.ArgumentParser(description='Train Medical ViT')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--data_path', type=str, default='data/chest_xray',
                       help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🏥 MEDICAL VISION TRANSFORMER TRAINING")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("="*60)
    
    # Load and update config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with command line arguments
    config['data']['dataset_path'] = args.data_path
    config['training']['epochs'] = args.epochs
    config['data']['batch_size'] = args.batch_size
    config['training']['learning_rate'] = args.lr
    
    # Save updated config
    with open(args.config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Configuration loaded from: {args.config}")
    print(f"Dataset path: {args.data_path}")
    print(f"Training epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("-"*60)
    
    try:
        # Train the model
        trainer, test_loader, class_names = train_model(args.config)
        
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Evaluate the trained model
        print("\nEvaluating trained model...")
        model_path = 'models/best_model.pth'
        if os.path.exists(model_path):
            metrics = evaluate_trained_model(model_path, args.config, test_loader, class_names)
            
            print("\n" + "="*60)
            print("📊 FINAL EVALUATION RESULTS")
            print("="*60)
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"{metric.upper()}: {value:.4f}")
            print("="*60)
        
        print(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("You can now run the Streamlit app with: streamlit run app.py")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
