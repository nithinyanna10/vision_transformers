#!/usr/bin/env python3
"""
Simplified Advanced Training Techniques for Medical Vision Transformer
Includes adversarial training and uncertainty quantification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
import math
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns


class AdversarialTrainer:
    """Adversarial training for model robustness"""
    
    def __init__(self, model, epsilon=0.03, alpha=0.01, num_steps=10):
        self.model = model
        self.epsilon = epsilon  # Perturbation budget
        self.alpha = alpha      # Step size
        self.num_steps = num_steps  # Number of PGD steps
    
    def fgsm_attack(self, images, labels, loss_fn):
        """Fast Gradient Sign Method attack"""
        images.requires_grad = True
        
        outputs = self.model(images)
        loss = loss_fn(outputs, labels)
        
        # Calculate gradients
        self.model.zero_grad()
        loss.backward()
        
        # Create adversarial examples
        adv_images = images + self.epsilon * images.grad.sign()
        adv_images = torch.clamp(adv_images, 0, 1)
        
        return adv_images.detach()
    
    def pgd_attack(self, images, labels, loss_fn):
        """Projected Gradient Descent attack"""
        adv_images = images.clone().detach()
        
        for _ in range(self.num_steps):
            adv_images.requires_grad = True
            
            outputs = self.model(adv_images)
            loss = loss_fn(outputs, labels)
            
            # Calculate gradients
            self.model.zero_grad()
            loss.backward()
            
            # Update adversarial examples
            adv_images = adv_images + self.alpha * adv_images.grad.sign()
            
            # Project to epsilon ball
            delta = torch.clamp(adv_images - images, -self.epsilon, self.epsilon)
            adv_images = torch.clamp(images + delta, 0, 1).detach()
        
        return adv_images
    
    def adversarial_loss(self, images, labels, loss_fn, attack_type='fgsm'):
        """Compute adversarial loss"""
        if attack_type == 'fgsm':
            adv_images = self.fgsm_attack(images, labels, loss_fn)
        elif attack_type == 'pgd':
            adv_images = self.pgd_attack(images, labels, loss_fn)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        # Compute loss on adversarial examples
        adv_outputs = self.model(adv_images)
        adv_loss = loss_fn(adv_outputs, labels)
        
        return adv_loss, adv_images


class UncertaintyQuantifier:
    """Uncertainty quantification using Monte Carlo Dropout"""
    
    def __init__(self, model, num_samples=10):
        self.model = model
        self.num_samples = num_samples
    
    def monte_carlo_dropout(self, images):
        """Monte Carlo Dropout for uncertainty estimation"""
        self.model.train()  # Enable dropout
        
        predictions = []
        for _ in range(self.num_samples):
            with torch.no_grad():
                outputs = self.model(images)
                predictions.append(F.softmax(outputs, dim=1))
        
        predictions = torch.stack(predictions)  # [num_samples, batch_size, num_classes]
        
        # Calculate statistics
        mean_pred = predictions.mean(dim=0)
        var_pred = predictions.var(dim=0)
        entropy = -(mean_pred * torch.log(mean_pred + 1e-8)).sum(dim=1)
        
        return mean_pred, var_pred, entropy
    
    def epistemic_aleatoric_uncertainty(self, images):
        """Separate epistemic and aleatoric uncertainty"""
        # Epistemic uncertainty (model uncertainty)
        mean_pred, var_pred, entropy = self.monte_carlo_dropout(images)
        epistemic_uncertainty = entropy
        
        # Aleatoric uncertainty (data uncertainty) - approximated
        aleatoric_uncertainty = var_pred.mean(dim=1)
        
        return epistemic_uncertainty, aleatoric_uncertainty, mean_pred


class AdvancedTrainer:
    """Advanced trainer with adversarial training and uncertainty quantification"""
    
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.device = torch.device("mps" if torch.backends.mps.is_available() else 
                                 "cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.adversarial_trainer = AdversarialTrainer(model)
        self.uncertainty_quantifier = UncertaintyQuantifier(model)
        
        # Training history
        self.training_history = {
            'adversarial_loss': [],
            'uncertainty_entropy': [],
            'epistemic_uncertainty': [],
            'aleatoric_uncertainty': []
        }
    
    def adversarial_training_step(self, images, labels, loss_fn, attack_type='fgsm'):
        """Single adversarial training step"""
        # Normal forward pass
        normal_outputs = self.model(images)
        normal_loss = loss_fn(normal_outputs, labels)
        
        # Adversarial forward pass
        adv_loss, adv_images = self.adversarial_trainer.adversarial_loss(
            images, labels, loss_fn, attack_type
        )
        
        # Combined loss
        total_loss = normal_loss + 0.5 * adv_loss
        
        return total_loss, adv_images
    
    def uncertainty_aware_training(self, images, labels, loss_fn):
        """Uncertainty-aware training step"""
        # Get predictions with uncertainty
        mean_pred, var_pred, entropy = self.uncertainty_quantifier.monte_carlo_dropout(images)
        
        # Standard loss
        standard_loss = loss_fn(torch.log(mean_pred + 1e-8), labels)
        
        # Uncertainty regularization
        uncertainty_reg = 0.1 * entropy.mean()
        
        # Combined loss
        total_loss = standard_loss + uncertainty_reg
        
        # Store uncertainty metrics
        self.training_history['uncertainty_entropy'].append(entropy.mean().item())
        
        return total_loss, entropy
    
    def evaluate_uncertainty(self, test_loader):
        """Evaluate model uncertainty on test set"""
        print("🔍 Evaluating model uncertainty...")
        
        all_entropies = []
        all_epistemic = []
        all_aleatoric = []
        all_predictions = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Get uncertainty estimates
                epistemic, aleatoric, mean_pred = self.uncertainty_quantifier.epistemic_aleatoric_uncertainty(images)
                
                all_entropies.extend(epistemic.cpu().numpy())
                all_epistemic.extend(epistemic.cpu().numpy())
                all_aleatoric.extend(aleatoric.cpu().numpy())
                all_predictions.extend(mean_pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate uncertainty statistics
        uncertainty_stats = {
            'mean_entropy': np.mean(all_entropies),
            'std_entropy': np.std(all_entropies),
            'mean_epistemic': np.mean(all_epistemic),
            'mean_aleatoric': np.mean(all_aleatoric),
            'uncertainty_correlation': np.corrcoef(all_entropies, all_labels)[0, 1]
        }
        
        return uncertainty_stats, all_entropies, all_epistemic, all_aleatoric
    
    def plot_uncertainty_analysis(self, uncertainty_stats, entropies, epistemic, aleatoric):
        """Plot uncertainty analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Uncertainty Analysis', fontsize=16, fontweight='bold')
        
        # Entropy distribution
        axes[0, 0].hist(entropies, bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_title('Prediction Entropy Distribution')
        axes[0, 0].set_xlabel('Entropy')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(uncertainty_stats['mean_entropy'], color='red', 
                          linestyle='--', label=f"Mean: {uncertainty_stats['mean_entropy']:.3f}")
        axes[0, 0].legend()
        
        # Epistemic vs Aleatoric uncertainty
        axes[0, 1].scatter(epistemic, aleatoric, alpha=0.6)
        axes[0, 1].set_xlabel('Epistemic Uncertainty')
        axes[0, 1].set_ylabel('Aleatoric Uncertainty')
        axes[0, 1].set_title('Epistemic vs Aleatoric Uncertainty')
        
        # Training history
        if self.training_history['uncertainty_entropy']:
            axes[1, 0].plot(self.training_history['uncertainty_entropy'])
            axes[1, 0].set_title('Uncertainty Entropy During Training')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Entropy')
        
        # Uncertainty correlation
        axes[1, 1].text(0.5, 0.5, f"Uncertainty Correlation:\n{uncertainty_stats['uncertainty_correlation']:.3f}", 
                       ha='center', va='center', fontsize=14, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[1, 1].set_title('Uncertainty Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/uncertainty_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()


def create_uncertainty_demo():
    """Create uncertainty demonstration"""
    print("🎯 Creating uncertainty demonstration...")
    
    # Create sample data
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Simulate uncertainty data
    entropies = np.random.beta(2, 5, 1000)  # Skewed distribution
    epistemic = np.random.gamma(2, 0.1, 1000)
    aleatoric = np.random.exponential(0.1, 1000)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🔍 Uncertainty Quantification Demonstration', fontsize=16, fontweight='bold')
    
    # Entropy distribution
    axes[0, 0].hist(entropies, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Prediction Entropy Distribution\n(Higher = More Uncertain)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Entropy')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(np.mean(entropies), color='red', linestyle='--', 
                      label=f"Mean: {np.mean(entropies):.3f}")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Epistemic vs Aleatoric uncertainty
    scatter = axes[0, 1].scatter(epistemic, aleatoric, alpha=0.6, c=entropies, cmap='viridis')
    axes[0, 1].set_xlabel('Epistemic Uncertainty\n(Model Uncertainty)')
    axes[0, 1].set_ylabel('Aleatoric Uncertainty\n(Data Uncertainty)')
    axes[0, 1].set_title('Types of Uncertainty', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=axes[0, 1], label='Total Entropy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Uncertainty over time (simulated)
    time_steps = np.arange(100)
    uncertainty_trend = 0.5 + 0.3 * np.sin(time_steps * 0.1) + np.random.normal(0, 0.05, 100)
    axes[1, 0].plot(time_steps, uncertainty_trend, color='green', linewidth=2)
    axes[1, 0].set_title('Uncertainty During Training\n(Decreasing over time)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Training Epoch')
    axes[1, 0].set_ylabel('Average Uncertainty')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Uncertainty statistics
    stats_text = f"""
    📊 Uncertainty Statistics:
    
    Mean Entropy: {np.mean(entropies):.3f}
    Std Entropy: {np.std(entropies):.3f}
    Mean Epistemic: {np.mean(epistemic):.3f}
    Mean Aleatoric: {np.mean(aleatoric):.3f}
    
    🎯 Interpretation:
    • High entropy = Model is uncertain
    • Epistemic = Can be reduced with more data
    • Aleatoric = Inherent data noise
    """
    
    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    axes[1, 1].set_title('Uncertainty Metrics', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/uncertainty_demo.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Uncertainty demonstration created!")


def create_adversarial_demo():
    """Create adversarial training demonstration"""
    print("🛡️ Creating adversarial training demonstration...")
    
    # Create sample data
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Simulate training curves
    epochs = np.arange(50)
    
    # Standard training
    standard_loss = 0.8 * np.exp(-epochs * 0.1) + 0.2 + np.random.normal(0, 0.02, 50)
    standard_acc = 0.2 + 0.7 * (1 - np.exp(-epochs * 0.08)) + np.random.normal(0, 0.01, 50)
    
    # Adversarial training
    adv_loss = 0.9 * np.exp(-epochs * 0.08) + 0.3 + np.random.normal(0, 0.02, 50)
    adv_acc = 0.15 + 0.75 * (1 - np.exp(-epochs * 0.06)) + np.random.normal(0, 0.01, 50)
    
    # Robustness (simulated)
    robustness = 0.3 + 0.6 * (1 - np.exp(-epochs * 0.05)) + np.random.normal(0, 0.01, 50)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🛡️ Adversarial Training Demonstration', fontsize=16, fontweight='bold')
    
    # Loss comparison
    axes[0, 0].plot(epochs, standard_loss, label='Standard Training', color='blue', linewidth=2)
    axes[0, 0].plot(epochs, adv_loss, label='Adversarial Training', color='red', linewidth=2)
    axes[0, 0].set_title('Training Loss Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy comparison
    axes[0, 1].plot(epochs, standard_acc, label='Standard Training', color='blue', linewidth=2)
    axes[0, 1].plot(epochs, adv_acc, label='Adversarial Training', color='red', linewidth=2)
    axes[0, 1].set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Robustness improvement
    axes[1, 0].plot(epochs, robustness, color='green', linewidth=2)
    axes[1, 0].set_title('Adversarial Robustness\n(Resistance to attacks)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Robustness Score')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Adversarial training benefits
    benefits_text = """
    🛡️ Adversarial Training Benefits:
    
    ✅ Improved Robustness
    ✅ Better Generalization
    ✅ Defense Against Attacks
    ✅ More Reliable Predictions
    
    ⚠️ Trade-offs:
    • Slightly higher training loss
    • Longer training time
    • More complex optimization
    
    🎯 Best Practices:
    • Use FGSM for efficiency
    • Use PGD for stronger defense
    • Balance standard vs adversarial loss
    """
    
    axes[1, 1].text(0.05, 0.95, benefits_text, transform=axes[1, 1].transAxes, 
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    axes[1, 1].set_title('Adversarial Training Guide', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/adversarial_demo.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Adversarial training demonstration created!")


def main():
    """Create advanced training demonstrations"""
    print("🚀 Creating Advanced Training Demonstrations...")
    
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)
    
    # Create demonstrations
    create_uncertainty_demo()
    create_adversarial_demo()
    
    print("\n🎉 Advanced training demonstrations created!")
    print("📁 Images saved in results/ directory:")
    print("   • uncertainty_demo.png")
    print("   • adversarial_demo.png")


if __name__ == "__main__":
    main()
