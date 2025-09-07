#!/usr/bin/env python3
"""
Advanced Model Architectures for Medical Vision Transformer
Includes hybrid CNN-ViT, ensemble methods, and multi-scale architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math
from typing import List, Dict, Optional, Tuple
import timm


class HybridCNNViT(nn.Module):
    """
    Hybrid CNN-ViT architecture combining ResNet backbone with Vision Transformer
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # CNN Backbone (ResNet50)
        self.cnn_backbone = models.resnet50(pretrained=True)
        self.cnn_backbone = nn.Sequential(*list(self.cnn_backbone.children())[:-2])  # Remove avgpool and fc
        
        # ViT Components
        self.patch_size = config['model']['patch_size']
        self.embed_dim = config['model']['embed_dim']
        self.num_heads = config['model']['num_heads']
        self.num_layers = config['model']['num_layers']
        self.img_size = config['model']['img_size']
        self.num_patches = (self.img_size // self.patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads, config['model']['mlp_ratio'], config['model']['dropout'])
            for _ in range(self.num_layers)
        ])
        
        # Cross-attention between CNN and ViT features
        self.cross_attention = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)
        
        # Feature fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(config['model']['dropout'])
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.LayerNorm(self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.embed_dim // 2, config['model']['num_classes'])
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # CNN feature extraction
        cnn_features = self.cnn_backbone(x)  # [B, 2048, 7, 7]
        cnn_features = F.adaptive_avg_pool2d(cnn_features, (1, 1)).flatten(1)  # [B, 2048]
        cnn_features = F.linear(cnn_features, self.embed_dim)  # [B, embed_dim]
        
        # ViT processing
        # Patch embedding
        patches = self.patch_embed(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        patches = patches.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Add CLS token
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        patches = torch.cat([cls_token, patches], dim=1)
        
        # Add positional embedding
        patches += self.pos_embed
        
        # Transformer blocks
        for transformer in self.transformer_blocks:
            patches = transformer(patches)
        
        # Extract CLS token
        vit_features = patches[:, 0]  # [B, embed_dim]
        
        # Cross-attention between CNN and ViT features
        cnn_features_expanded = cnn_features.unsqueeze(1)  # [B, 1, embed_dim]
        vit_features_expanded = vit_features.unsqueeze(1)  # [B, 1, embed_dim]
        
        attended_features, _ = self.cross_attention(
            vit_features_expanded, cnn_features_expanded, cnn_features_expanded
        )
        attended_features = attended_features.squeeze(1)  # [B, embed_dim]
        
        # Feature fusion
        fused_features = torch.cat([vit_features, attended_features], dim=1)
        fused_features = self.fusion_layer(fused_features)
        
        # Classification
        output = self.classifier(fused_features)
        return output


class MultiScaleViT(nn.Module):
    """
    Multi-scale Vision Transformer with different patch sizes
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Multiple patch sizes for different scales
        self.patch_sizes = [8, 16, 32]
        self.embed_dim = config['model']['embed_dim']
        self.num_classes = config['model']['num_classes']
        
        # Create ViT branches for different scales
        self.vit_branches = nn.ModuleList()
        for patch_size in self.patch_sizes:
            branch = ViTBranch(patch_size, config)
            self.vit_branches.append(branch)
        
        # Feature fusion
        total_dim = self.embed_dim * len(self.patch_sizes)
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(config['model']['dropout'])
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.LayerNorm(self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.embed_dim // 2, self.num_classes)
        )
    
    def forward(self, x):
        # Process through different scales
        scale_features = []
        for branch in self.vit_branches:
            features = branch(x)
            scale_features.append(features)
        
        # Concatenate features from all scales
        combined_features = torch.cat(scale_features, dim=1)
        
        # Fusion and classification
        fused_features = self.fusion(combined_features)
        output = self.classifier(fused_features)
        
        return output


class ViTBranch(nn.Module):
    """Individual ViT branch for multi-scale architecture"""
    def __init__(self, patch_size: int, config: Dict):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = config['model']['embed_dim']
        self.num_heads = config['model']['num_heads']
        self.num_layers = config['model']['num_layers']
        self.img_size = config['model']['img_size']
        self.num_patches = (self.img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, self.embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads, config['model']['mlp_ratio'], config['model']['dropout'])
            for _ in range(self.num_layers // 2)  # Fewer layers per branch
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Patch embedding
        patches = self.patch_embed(x)
        patches = patches.flatten(2).transpose(1, 2)
        
        # Add CLS token
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        patches = torch.cat([cls_token, patches], dim=1)
        
        # Add positional embedding
        patches += self.pos_embed
        
        # Transformer blocks
        for transformer in self.transformer_blocks:
            patches = transformer(patches)
        
        # Return CLS token
        return patches[:, 0]


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models for improved performance
    """
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.weights = torch.tensor(self.weights, dtype=torch.float32)
        self.weights = self.weights / self.weights.sum()  # Normalize weights
    
    def forward(self, x):
        predictions = []
        for model in self.models:
            pred = F.softmax(model(x), dim=1)
            predictions.append(pred)
        
        # Weighted average of predictions
        ensemble_pred = torch.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            ensemble_pred += self.weights[i] * pred
        
        return torch.log(ensemble_pred + 1e-8)  # Log for cross-entropy loss


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


def create_advanced_model(model_type: str, config: Dict) -> nn.Module:
    """Factory function to create advanced models"""
    if model_type == "hybrid_cnn_vit":
        return HybridCNNViT(config)
    elif model_type == "multiscale_vit":
        return MultiScaleViT(config)
    elif model_type == "ensemble":
        # Create ensemble of different models
        models_list = [
            HybridCNNViT(config),
            MultiScaleViT(config),
            # Add more models as needed
        ]
        return EnsembleModel(models_list)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_pretrained_models(config: Dict) -> Dict[str, nn.Module]:
    """Load multiple pretrained models for ensemble"""
    models = {}
    
    # Load our trained ViT
    try:
        vit_model = create_model(config)  # Original ViT
        checkpoint = torch.load('models/best_model.pth', map_location='cpu', weights_only=False)
        vit_model.load_state_dict(checkpoint['model_state_dict'])
        models['vit'] = vit_model
    except:
        print("Warning: Could not load trained ViT model")
    
    # Load pretrained EfficientNet
    try:
        efficientnet = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
        models['efficientnet'] = efficientnet
    except:
        print("Warning: Could not load EfficientNet")
    
    # Load pretrained DenseNet
    try:
        densenet = models.densenet121(pretrained=True)
        densenet.classifier = nn.Linear(densenet.classifier.in_features, 2)
        models['densenet'] = densenet
    except:
        print("Warning: Could not load DenseNet")
    
    return models


if __name__ == "__main__":
    # Test the advanced models
    config = {
        'model': {
            'patch_size': 16,
            'embed_dim': 512,
            'num_heads': 8,
            'num_layers': 12,
            'img_size': 224,
            'mlp_ratio': 4.0,
            'dropout': 0.1,
            'num_classes': 2
        }
    }
    
    # Test Hybrid CNN-ViT
    hybrid_model = HybridCNNViT(config)
    test_input = torch.randn(2, 3, 224, 224)
    output = hybrid_model(test_input)
    print(f"Hybrid CNN-ViT output shape: {output.shape}")
    
    # Test Multi-scale ViT
    multiscale_model = MultiScaleViT(config)
    output = multiscale_model(test_input)
    print(f"Multi-scale ViT output shape: {output.shape}")
    
    print("✅ Advanced models created successfully!")
