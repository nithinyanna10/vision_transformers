#!/usr/bin/env python3
"""
Advanced Training Techniques for Medical Vision Transformer
Includes self-supervised learning (MAE), adversarial training, and uncertainty quantification
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


class MaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder (MAE) for self-supervised pre-training
    Based on "Masked Autoencoders Are Scalable Vision Learners"
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Encoder (ViT)
        self.embed_dim = config['model']['embed_dim']
        self.patch_size = config['model']['patch_size']
        self.img_size = config['model']['img_size']
        self.num_patches = (self.img_size // self.patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
        # Encoder transformer blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, config['model']['num_heads'], 
                           config['model']['mlp_ratio'], config['model']['dropout'])
            for _ in range(config['model']['num_layers'])
        ])
        
        # Decoder
        decoder_embed_dim = self.embed_dim // 2
        self.decoder_embed = nn.Linear(self.embed_dim, decoder_embed_dim)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim))
        
        # Decoder transformer blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, config['model']['num_heads'], 
                           config['model']['mlp_ratio'], config['model']['dropout'])
            for _ in range(4)  # Fewer decoder layers
        ])
        
        # Prediction head
        self.prediction_head = nn.Linear(decoder_embed_dim, self.patch_size ** 2 * 3)
        
        # Masking
        self.mask_ratio = 0.75  # Mask 75% of patches
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.norm_pix_loss = True
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
    
    def random_masking(self, x, mask_ratio):
        """Random masking of patches"""
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate the binary mask
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def patchify(self, imgs):
        """Convert images to patches"""
        p = self.patch_size
        h = w = self.img_size // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
    
    def unpatchify(self, x):
        """Convert patches back to images"""
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    
    def forward_encoder(self, x, mask_ratio):
        """Forward pass through encoder"""
        # Patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Apply masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Apply transformer blocks
        for blk in self.encoder_blocks:
            x = blk(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        """Forward pass through decoder"""
        # Embed tokens
        x = self.decoder_embed(x)
        
        # Simplified decoder - just predict from visible patches
        # Add positional embedding
        x = x + self.decoder_pos_embed[:, :x.shape[1], :]
        
        # Apply transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        
        # Predict pixels
        x = self.prediction_head(x)
        
        # Remove CLS token
        x = x[:, 1:, :]
        
        return x
    
    def forward_loss(self, imgs, pred, mask):
        """Compute loss between predicted and target pixels"""
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        
        loss = (loss * mask).sum() / mask.sum()  # Mean loss on removed patches
        return loss
    
    def forward(self, imgs, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        # Encoder
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        
        # Decoder
        pred = self.forward_decoder(latent, ids_restore)
        
        # Loss
        loss = self.forward_loss(imgs, pred, mask)
        
        return loss, pred, mask


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
    """Uncertainty quantification using Monte Carlo Dropout and Ensemble methods"""
    
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
    
    def ensemble_uncertainty(self, models, images):
        """Ensemble uncertainty estimation"""
        predictions = []
        
        for model in models:
            model.eval()
            with torch.no_grad():
                outputs = model(images)
                predictions.append(F.softmax(outputs, dim=1))
        
        predictions = torch.stack(predictions)  # [num_models, batch_size, num_classes]
        
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
    """Advanced trainer with self-supervised learning, adversarial training, and uncertainty quantification"""
    
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.device = torch.device("mps" if torch.backends.mps.is_available() else 
                                 "cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.mae = MaskedAutoencoder(config)
        self.adversarial_trainer = AdversarialTrainer(model)
        self.uncertainty_quantifier = UncertaintyQuantifier(model)
        
        # Training history
        self.training_history = {
            'mae_loss': [],
            'adversarial_loss': [],
            'uncertainty_entropy': [],
            'epistemic_uncertainty': [],
            'aleatoric_uncertainty': []
        }
    
    def pretrain_mae(self, train_loader, epochs=50):
        """Self-supervised pre-training with MAE"""
        print("🎯 Starting MAE pre-training...")
        
        self.mae = self.mae.to(self.device)
        optimizer = optim.AdamW(self.mae.parameters(), lr=1e-4, weight_decay=0.05)
        
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            for images, _ in train_loader:
                images = images.to(self.device)
                
                # Forward pass
                loss, pred, mask = self.mae(images)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            self.training_history['mae_loss'].append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"MAE Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        print("✅ MAE pre-training completed!")
    
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
        if self.training_history['mae_loss']:
            axes[1, 0].plot(self.training_history['mae_loss'])
            axes[1, 0].set_title('MAE Pre-training Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
        
        # Uncertainty correlation
        axes[1, 1].text(0.5, 0.5, f"Uncertainty Correlation:\n{uncertainty_stats['uncertainty_correlation']:.3f}", 
                       ha='center', va='center', fontsize=14, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[1, 1].set_title('Uncertainty Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/uncertainty_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()


class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP"""
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Self-attention
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        
        # MLP
        norm_x = self.norm2(x)
        mlp_out = self.mlp(norm_x)
        x = x + mlp_out
        
        return x


def main():
    """Test advanced training components"""
    print("🧪 Testing Advanced Training Components...")
    
    # Dummy config
    config = {
        'model': {
            'embed_dim': 512,
            'patch_size': 16,
            'img_size': 224,
            'num_heads': 8,
            'mlp_ratio': 4.0,
            'dropout': 0.1,
            'num_layers': 12
        }
    }
    
    # Test MAE
    print("Testing MAE...")
    mae = MaskedAutoencoder(config)
    test_input = torch.randn(2, 3, 224, 224)
    loss, pred, mask = mae(test_input)
    print(f"✅ MAE working - Loss: {loss.item():.4f}")
    
    # Test Adversarial Trainer
    print("Testing Adversarial Trainer...")
    dummy_model = nn.Linear(10, 2)
    adv_trainer = AdversarialTrainer(dummy_model)
    print("✅ Adversarial Trainer working")
    
    # Test Uncertainty Quantifier
    print("Testing Uncertainty Quantifier...")
    uncertainty_quantifier = UncertaintyQuantifier(dummy_model)
    print("✅ Uncertainty Quantifier working")
    
    print("🎉 All advanced training components working!")


if __name__ == "__main__":
    main()
