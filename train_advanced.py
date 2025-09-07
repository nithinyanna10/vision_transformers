#!/usr/bin/env python3
"""
Advanced Training Script for Medical Vision Transformer
Supports hybrid architectures, ensemble methods, and advanced training techniques
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from src.advanced_models import create_advanced_model, load_pretrained_models, EnsembleModel
from src.data_loader import DataManager
from src.evaluator import evaluate_model
import timm


class AdvancedTrainer:
    """Advanced trainer with support for multiple architectures"""
    
    def __init__(self, config, model_type="hybrid_cnn_vit"):
        self.config = config
        self.model_type = model_type
        self.device = self._get_device()
        
        # Create model
        if model_type == "ensemble":
            self.model = self._create_ensemble_model()
        else:
            self.model = create_advanced_model(model_type, config)
        
        self.model = self.model.to(self.device)
        
        # Setup training components
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.criterion = self._setup_criterion()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
    def _get_device(self):
        """Get the best available device"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _create_ensemble_model(self):
        """Create ensemble model from pretrained models"""
        models_dict = load_pretrained_models(self.config)
        models_list = list(models_dict.values())
        
        if len(models_list) == 0:
            raise ValueError("No pretrained models available for ensemble")
        
        return EnsembleModel(models_list)
    
    def _setup_optimizer(self):
        """Setup optimizer with different learning rates for different parts"""
        if self.model_type == "ensemble":
            # For ensemble, use lower learning rate
            return optim.AdamW(self.model.parameters(), lr=1e-5, weight_decay=0.01)
        else:
            # Different learning rates for different parts
            backbone_params = []
            head_params = []
            
            for name, param in self.model.named_parameters():
                if 'backbone' in name or 'cnn_backbone' in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)
            
            return optim.AdamW([
                {'params': backbone_params, 'lr': 1e-5},  # Lower LR for pretrained parts
                {'params': head_params, 'lr': 1e-4}       # Higher LR for new parts
            ], weight_decay=0.01)
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-7
        )
    
    def _setup_criterion(self):
        """Setup loss function"""
        # Focal loss for handling class imbalance
        class FocalLoss(nn.Module):
            def __init__(self, alpha=0.25, gamma=2.0):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
            
            def forward(self, inputs, targets):
                ce_loss = F.cross_entropy(inputs, targets, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                return focal_loss.mean()
        
        return FocalLoss(alpha=0.25, gamma=2.0)
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=20):
        """Main training loop"""
        print(f"🚀 Starting training with {self.model_type} architecture")
        print(f"📱 Device: {self.device}")
        print(f"🔢 Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            print(f"\n📅 Epoch {epoch+1}/{epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self.save_model(f"models/best_{self.model_type}_model.pth")
                print(f"💾 New best model saved! Val Acc: {val_acc:.2f}%")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config['training']['early_stopping_patience']:
                print(f"🛑 Early stopping after {epoch+1} epochs")
                break
        
        # Plot training curves
        self.plot_training_curves()
        
        print(f"\n🎉 Training completed! Best validation accuracy: {self.best_val_acc:.2f}%")
    
    def save_model(self, path):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config,
            'model_type': self.model_type
        }, path)
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Val Loss', color='red')
        ax1.set_title(f'{self.model_type.title()} - Training Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.train_accuracies, label='Train Acc', color='blue')
        ax2.plot(self.val_accuracies, label='Val Acc', color='red')
        ax2.set_title(f'{self.model_type.title()} - Accuracy Curves')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'models/{self.model_type}_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train Advanced Medical ViT Models')
    parser.add_argument('--model_type', type=str, default='hybrid_cnn_vit',
                       choices=['hybrid_cnn_vit', 'multiscale_vit', 'ensemble'],
                       help='Type of advanced model to train')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with command line arguments
    config['training']['epochs'] = args.epochs
    config['data']['batch_size'] = args.batch_size
    
    # Create data loaders
    print("📊 Loading data...")
    data_manager = DataManager(config)
    train_loader, val_loader, test_loader, class_names = data_manager.create_dataloaders()
    
    # Create trainer
    trainer = AdvancedTrainer(config, args.model_type)
    
    # Train model
    trainer.train(train_loader, val_loader, args.epochs)
    
    # Evaluate on test set
    print("\n🧪 Evaluating on test set...")
    test_metrics = evaluate_model(trainer.model, test_loader, class_names, trainer.device)
    
    print(f"\n📊 Final Test Results for {args.model_type}:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
