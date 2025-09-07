"""
Advanced Training Pipeline for Medical ViT
Includes mixed precision, learning rate scheduling, early stopping, and monitoring
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import yaml
from datetime import datetime
import json
from src.model import FocalLoss
from src.data_loader import mixup_data, mixup_criterion


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


class LearningRateScheduler:
    """Advanced learning rate scheduling"""
    
    def __init__(self, optimizer, config):
        self.optimizer = optimizer
        self.config = config
        self.warmup_epochs = config['training']['warmup_epochs']
        self.total_epochs = config['training']['epochs']
        self.base_lr = config['training']['learning_rate']
        
    def step(self, epoch, val_loss=None):
        if epoch < self.warmup_epochs:
            # Warmup phase
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr


class MetricsTracker:
    """Track and visualize training metrics"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        
    def update(self, train_loss, val_loss, train_acc, val_acc, lr):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
    
    def plot_metrics(self, save_path=None):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.train_accuracies, label='Train Accuracy')
        axes[0, 1].plot(self.val_accuracies, label='Val Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[1, 0].plot(self.learning_rates)
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
        
        # Combined plot
        ax2 = axes[1, 1].twinx()
        axes[1, 1].plot(self.train_losses, 'b-', label='Train Loss')
        axes[1, 1].plot(self.val_losses, 'r-', label='Val Loss')
        ax2.plot(self.train_accuracies, 'g--', label='Train Acc')
        ax2.plot(self.val_accuracies, 'm--', label='Val Acc')
        
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss', color='b')
        ax2.set_ylabel('Accuracy', color='g')
        axes[1, 1].set_title('Combined Metrics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class MedicalViTTrainer:
    """Main trainer class for Medical ViT"""
    
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        
        # Initialize components
        self.setup_loss_function()
        self.setup_optimizer()
        self.setup_scheduler()
        self.setup_early_stopping()
        self.setup_metrics_tracker()
        
        # Mixed precision training (only for CUDA)
        if device.type == 'cuda':
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Create save directory
        self.save_dir = config['paths']['model_save_path']
        os.makedirs(self.save_dir, exist_ok=True)
        
    def setup_loss_function(self):
        """Setup loss function based on config"""
        loss_config = self.config['loss']
        
        if loss_config['type'] == 'focal':
            self.criterion = FocalLoss(
                alpha=loss_config['alpha'],
                gamma=loss_config['gamma']
            )
        elif loss_config['type'] == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=loss_config.get('label_smoothing', 0.0)
            )
        else:
            self.criterion = nn.CrossEntropyLoss()
    
    def setup_optimizer(self):
        """Setup optimizer with weight decay"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
    
    def setup_scheduler(self):
        """Setup learning rate scheduler"""
        self.scheduler = LearningRateScheduler(self.optimizer, self.config)
    
    def setup_early_stopping(self):
        """Setup early stopping"""
        self.early_stopping = EarlyStopping(
            patience=self.config['training']['early_stopping_patience']
        )
    
    def setup_metrics_tracker(self):
        """Setup metrics tracker"""
        self.metrics_tracker = MetricsTracker()
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Mixup augmentation
            if self.config['augmentation'].get('mixup_alpha', 0) > 0 and np.random.random() < 0.5:
                data, target_a, target_b, lam = mixup_data(
                    data, target, self.config['augmentation']['mixup_alpha']
                )
                
                self.optimizer.zero_grad()
                
                if self.scaler:
                    with autocast():
                        output = self.model(data)
                        loss = mixup_criterion(self.criterion, output, target_a, target_b, lam)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    output = self.model(data)
                    loss = mixup_criterion(self.criterion, output, target_a, target_b, lam)
                    
                    loss.backward()
                    self.optimizer.step()
                
                # For metrics, use original target
                with torch.no_grad():
                    pred = output.argmax(dim=1)
                    all_predictions.extend(pred.cpu().numpy())
                    all_targets.extend(target_a.cpu().numpy())
                    
            else:
                self.optimizer.zero_grad()
                
                if self.scaler:
                    with autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    
                    loss.backward()
                    self.optimizer.step()
                
                # Store predictions and targets
                with torch.no_grad():
                    pred = output.argmax(dim=1)
                    all_predictions.extend(pred.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                
                if self.scaler:
                    with autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probabilities = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        
        # Calculate additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted'
        )
        
        try:
            auc = roc_auc_score(all_targets, all_probabilities[:, 1])
        except:
            auc = 0.0
        
        return avg_loss, accuracy, precision, recall, f1, auc
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, 'best_model.pth'))
        
        torch.save(checkpoint, os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_acc = 0
        
        for epoch in range(self.config['training']['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['training']['epochs']}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, precision, recall, f1, auc = self.validate_epoch(val_loader)
            
            # Update learning rate
            current_lr = self.scheduler.step(epoch, val_loss)
            
            # Update metrics tracker
            self.metrics_tracker.update(train_loss, val_loss, train_acc, val_acc, current_lr)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(epoch, is_best=True)
                print(f"New best model saved! Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")
        
        # Plot and save metrics
        metrics_path = os.path.join(self.save_dir, 'training_metrics.png')
        self.metrics_tracker.plot_metrics(metrics_path)
        
        return best_val_acc


def train_model(config_path='configs/config.yaml'):
    """Main training function"""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    # Prefer Apple Silicon GPU (MPS) if available, else CUDA, else CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    from src.data_loader import DataManager
    data_manager = DataManager(config)
    train_loader, val_loader, test_loader, class_names = data_manager.create_dataloaders()
    
    # Create model
    from src.model import create_model
    model = create_model(config).to(device)
    
    # Create trainer
    trainer = MedicalViTTrainer(model, config, device)
    
    # Train model
    best_acc = trainer.train(train_loader, val_loader)
    
    return trainer, test_loader, class_names


if __name__ == "__main__":
    trainer, test_loader, class_names = train_model()
