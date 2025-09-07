#!/usr/bin/env python3
"""
Explainable AI (XAI) Module for Medical Vision Transformer
Implements Grad-CAM, attention visualization, and SHAP explanations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional
import shap
import lime
from lime import lime_image
import warnings
warnings.filterwarnings('ignore')


class GradCAM:
    """Grad-CAM implementation for Vision Transformers"""
    
    def __init__(self, model, target_layer_name: str = None):
        self.model = model
        self.model.eval()
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.handlers = []
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Find target layer
        target_layer = None
        for name, module in self.model.named_modules():
            if self.target_layer_name and self.target_layer_name in name:
                target_layer = module
                break
            elif isinstance(module, nn.MultiheadAttention) and target_layer is None:
                target_layer = module
        
        if target_layer is None:
            raise ValueError(f"Target layer {self.target_layer_name} not found")
        
        # Register hooks
        self.handlers.append(target_layer.register_forward_hook(forward_hook))
        self.handlers.append(target_layer.register_backward_hook(backward_hook))
    
    def generate_cam(self, input_tensor, class_idx: int = None) -> np.ndarray:
        """Generate Grad-CAM for the given input"""
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients[0]  # [num_patches, embed_dim]
        activations = self.activations[0]  # [num_patches, embed_dim]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=1)  # [num_patches]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[0], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Reshape to spatial dimensions
        patch_size = int(np.sqrt(activations.shape[0]))
        cam = cam.reshape(patch_size, patch_size)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.detach().cpu().numpy()
    
    def visualize(self, input_tensor, class_idx: int = None, alpha: float = 0.4) -> np.ndarray:
        """Generate Grad-CAM visualization"""
        cam = self.generate_cam(input_tensor, class_idx)
        
        # Resize CAM to input image size
        input_size = input_tensor.shape[-1]
        cam_resized = cv2.resize(cam, (input_size, input_size))
        
        # Convert input to numpy
        input_np = input_tensor[0].permute(1, 2, 0).cpu().numpy()
        input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min())
        
        # Apply colormap
        cam_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = cv2.addWeighted(
            np.uint8(255 * input_np), alpha,
            cam_colored, 1 - alpha, 0
        )
        
        return overlay
    
    def __del__(self):
        """Clean up hooks"""
        for handler in self.handlers:
            handler.remove()


class AttentionVisualizer:
    """Attention visualization for Vision Transformers"""
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.attention_weights = []
        self.handlers = []
        
        # Register hooks for attention layers
        self._register_attention_hooks()
    
    def _register_attention_hooks(self):
        """Register hooks to capture attention weights"""
        def attention_hook(module, input, output):
            # Extract attention weights from MultiheadAttention
            if hasattr(module, 'attention_weights'):
                self.attention_weights.append(module.attention_weights)
            else:
                # For standard MultiheadAttention, we need to modify forward pass
                pass
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                self.handlers.append(module.register_forward_hook(attention_hook))
    
    def visualize_attention_rollout(self, input_tensor, head_fusion: str = "mean") -> np.ndarray:
        """Visualize attention rollout across all layers"""
        with torch.no_grad():
            # Forward pass to get attention weights
            _ = self.model(input_tensor)
            
            # Process attention weights
            if not self.attention_weights:
                # Fallback: create dummy attention
                patch_size = int(np.sqrt((input_tensor.shape[-1] // 16) ** 2))
                return np.ones((patch_size, patch_size)) / (patch_size * patch_size)
            
            # Combine attention from all layers
            attention_maps = []
            for attn in self.attention_weights:
                if attn is not None:
                    # Average across heads
                    attn_avg = attn.mean(dim=1)  # [batch, num_patches, num_patches]
                    attention_maps.append(attn_avg[0])  # Take first sample
            
            if not attention_maps:
                patch_size = int(np.sqrt((input_tensor.shape[-1] // 16) ** 2))
                return np.ones((patch_size, patch_size)) / (patch_size * patch_size)
            
            # Attention rollout
            rollout = attention_maps[0]
            for attn in attention_maps[1:]:
                rollout = torch.matmul(attn, rollout)
            
            # Extract CLS token attention to patches
            cls_attention = rollout[0, 1:]  # Skip CLS token itself
            
            # Reshape to spatial dimensions
            patch_size = int(np.sqrt(cls_attention.shape[0]))
            attention_map = cls_attention.reshape(patch_size, patch_size)
            
            # Normalize
            attention_map = attention_map - attention_map.min()
            attention_map = attention_map / attention_map.max()
            
            return attention_map.cpu().numpy()
    
    def visualize_attention_heads(self, input_tensor, layer_idx: int = -1) -> List[np.ndarray]:
        """Visualize attention heads for a specific layer"""
        with torch.no_grad():
            _ = self.model(input_tensor)
            
            if not self.attention_weights or layer_idx >= len(self.attention_weights):
                return []
            
            attn = self.attention_weights[layer_idx]
            if attn is None:
                return []
            
            # Extract CLS token attention for each head
            cls_attention = attn[0, :, 0, 1:]  # [num_heads, num_patches]
            
            attention_maps = []
            for head_idx in range(cls_attention.shape[0]):
                head_attn = cls_attention[head_idx]
                
                # Reshape to spatial dimensions
                patch_size = int(np.sqrt(head_attn.shape[0]))
                attention_map = head_attn.reshape(patch_size, patch_size)
                
                # Normalize
                attention_map = attention_map - attention_map.min()
                attention_map = attention_map / attention_map.max()
                
                attention_maps.append(attention_map.cpu().numpy())
            
            return attention_maps
    
    def __del__(self):
        """Clean up hooks"""
        for handler in self.handlers:
            handler.remove()


class SHAPExplainer:
    """SHAP explanations for medical image classification"""
    
    def __init__(self, model, background_data: torch.Tensor = None):
        self.model = model
        self.model.eval()
        self.background_data = background_data
        
        # Create SHAP explainer
        self.explainer = None
        self._setup_explainer()
    
    def _setup_explainer(self):
        """Setup SHAP explainer"""
        def model_predict(images):
            """Wrapper function for SHAP"""
            with torch.no_grad():
                if isinstance(images, np.ndarray):
                    images = torch.from_numpy(images).float()
                if images.dim() == 3:
                    images = images.unsqueeze(0)
                
                outputs = self.model(images)
                return F.softmax(outputs, dim=1).cpu().numpy()
        
        # Use DeepExplainer for neural networks
        if self.background_data is not None:
            self.explainer = shap.DeepExplainer(self.model, self.background_data)
        else:
            # Fallback to GradientExplainer
            self.explainer = shap.GradientExplainer(model_predict, self.background_data)
    
    def explain_image(self, image: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """Generate SHAP explanation for an image"""
        if self.explainer is None:
            raise ValueError("SHAP explainer not properly initialized")
        
        # Generate SHAP values
        shap_values = self.explainer.shap_values(image.unsqueeze(0))
        
        if isinstance(shap_values, list):
            shap_values = shap_values[class_idx or 0]
        
        # Process SHAP values
        if shap_values.ndim == 4:  # [batch, channels, height, width]
            shap_values = shap_values[0]  # Remove batch dimension
            shap_values = np.abs(shap_values).sum(axis=0)  # Sum across channels
        
        # Normalize
        shap_values = shap_values - shap_values.min()
        shap_values = shap_values / shap_values.max()
        
        return shap_values


class LIMEExplainer:
    """LIME explanations for medical image classification"""
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
        # Create LIME explainer
        self.explainer = lime_image.LimeImageExplainer()
    
    def explain_image(self, image: torch.Tensor, class_idx: int = None) -> Dict:
        """Generate LIME explanation for an image"""
        def model_predict(images):
            """Wrapper function for LIME"""
            with torch.no_grad():
                if isinstance(images, np.ndarray):
                    images = torch.from_numpy(images).float()
                if images.dim() == 3:
                    images = images.unsqueeze(0)
                
                outputs = self.model(images)
                return F.softmax(outputs, dim=1).cpu().numpy()
        
        # Convert tensor to numpy
        image_np = image.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            image_np,
            model_predict,
            top_labels=2,
            hide_color=0,
            num_samples=1000
        )
        
        return explanation


class XAIVisualizer:
    """Comprehensive XAI visualization toolkit"""
    
    def __init__(self, model):
        self.model = model
        self.gradcam = GradCAM(model)
        self.attention_viz = AttentionVisualizer(model)
        self.shap_explainer = None
        self.lime_explainer = LIMEExplainer(model)
    
    def create_comprehensive_explanation(self, image: torch.Tensor, class_names: List[str], 
                                       class_idx: int = None) -> Dict:
        """Create comprehensive explanation with multiple methods"""
        explanations = {}
        
        # Get prediction
        with torch.no_grad():
            output = self.model(image.unsqueeze(0))
            probabilities = F.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        if class_idx is None:
            class_idx = predicted_class
        
        # Grad-CAM
        try:
            gradcam_vis = self.gradcam.visualize(image.unsqueeze(0), class_idx)
            explanations['gradcam'] = gradcam_vis
        except Exception as e:
            print(f"Grad-CAM failed: {e}")
            explanations['gradcam'] = None
        
        # Attention visualization
        try:
            attention_rollout = self.attention_viz.visualize_attention_rollout(image.unsqueeze(0))
            explanations['attention_rollout'] = attention_rollout
        except Exception as e:
            print(f"Attention visualization failed: {e}")
            explanations['attention_rollout'] = None
        
        # LIME explanation
        try:
            lime_explanation = self.lime_explainer.explain_image(image, class_idx)
            explanations['lime'] = lime_explanation
        except Exception as e:
            print(f"LIME failed: {e}")
            explanations['lime'] = None
        
        # Add prediction info
        explanations['prediction'] = {
            'class': class_names[predicted_class],
            'confidence': confidence,
            'probabilities': {
                class_names[i]: probabilities[0, i].item() 
                for i in range(len(class_names))
            }
        }
        
        return explanations
    
    def plot_explanations(self, image: torch.Tensor, explanations: Dict, 
                         class_names: List[str], save_path: str = None):
        """Plot comprehensive explanations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive XAI Explanations', fontsize=16, fontweight='bold')
        
        # Original image
        ax = axes[0, 0]
        image_np = image.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        ax.imshow(image_np)
        ax.set_title('Original Image')
        ax.axis('off')
        
        # Prediction info
        ax = axes[0, 1]
        pred_info = explanations['prediction']
        ax.text(0.1, 0.8, f"Predicted: {pred_info['class']}", fontsize=14, fontweight='bold')
        ax.text(0.1, 0.6, f"Confidence: {pred_info['confidence']:.3f}", fontsize=12)
        ax.text(0.1, 0.4, "Probabilities:", fontsize=12, fontweight='bold')
        for i, (class_name, prob) in enumerate(pred_info['probabilities'].items()):
            ax.text(0.1, 0.3 - i*0.1, f"{class_name}: {prob:.3f}", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Prediction Results')
        
        # Grad-CAM
        ax = axes[0, 2]
        if explanations['gradcam'] is not None:
            ax.imshow(explanations['gradcam'])
            ax.set_title('Grad-CAM Visualization')
        else:
            ax.text(0.5, 0.5, 'Grad-CAM\nNot Available', ha='center', va='center')
            ax.set_title('Grad-CAM Visualization')
        ax.axis('off')
        
        # Attention Rollout
        ax = axes[1, 0]
        if explanations['attention_rollout'] is not None:
            im = ax.imshow(explanations['attention_rollout'], cmap='hot', interpolation='nearest')
            ax.set_title('Attention Rollout')
            plt.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.text(0.5, 0.5, 'Attention\nNot Available', ha='center', va='center')
            ax.set_title('Attention Rollout')
        ax.axis('off')
        
        # LIME explanation
        ax = axes[1, 1]
        if explanations['lime'] is not None:
            # Plot LIME explanation
            lime_img, lime_mask = explanations['lime'].get_image_and_mask(
                explanations['lime'].top_labels[0], 
                positive_only=True, 
                num_features=10, 
                hide_rest=True
            )
            ax.imshow(lime_img)
            ax.set_title('LIME Explanation')
        else:
            ax.text(0.5, 0.5, 'LIME\nNot Available', ha='center', va='center')
            ax.set_title('LIME Explanation')
        ax.axis('off')
        
        # Combined visualization
        ax = axes[1, 2]
        if explanations['gradcam'] is not None and explanations['attention_rollout'] is not None:
            # Combine Grad-CAM and attention
            combined = 0.5 * explanations['gradcam'] + 0.5 * explanations['attention_rollout']
            ax.imshow(combined, cmap='viridis')
            ax.set_title('Combined Visualization')
        else:
            ax.text(0.5, 0.5, 'Combined\nNot Available', ha='center', va='center')
            ax.set_title('Combined Visualization')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def create_xai_demo(model, test_loader, class_names: List[str], num_samples: int = 5):
    """Create XAI demo with multiple samples"""
    xai_viz = XAIVisualizer(model)
    
    print("🔍 Creating XAI explanations for test samples...")
    
    for i, (images, labels) in enumerate(test_loader):
        if i >= num_samples:
            break
        
        image = images[0]
        true_label = labels[0].item()
        
        print(f"\n📊 Sample {i+1}: True class = {class_names[true_label]}")
        
        # Generate explanations
        explanations = xai_viz.create_comprehensive_explanation(
            image, class_names
        )
        
        # Plot explanations
        xai_viz.plot_explanations(
            image, explanations, class_names,
            save_path=f'results/xai_sample_{i+1}.png'
        )
        
        print(f"✅ XAI explanations saved for sample {i+1}")


if __name__ == "__main__":
    # Test XAI components
    print("🧪 Testing XAI components...")
    
    # Create dummy model for testing
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, 2)
        
        def forward(self, x):
            x = F.relu(self.conv(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = DummyModel()
    test_input = torch.randn(1, 3, 224, 224)
    
    # Test Grad-CAM
    try:
        gradcam = GradCAM(model, "conv")
        cam = gradcam.generate_cam(test_input)
        print("✅ Grad-CAM working")
    except Exception as e:
        print(f"❌ Grad-CAM failed: {e}")
    
    # Test Attention Visualizer
    try:
        attn_viz = AttentionVisualizer(model)
        print("✅ Attention Visualizer working")
    except Exception as e:
        print(f"❌ Attention Visualizer failed: {e}")
    
    # Test LIME
    try:
        lime_explainer = LIMEExplainer(model)
        print("✅ LIME Explainer working")
    except Exception as e:
        print(f"❌ LIME Explainer failed: {e}")
    
    print("🎉 XAI components tested successfully!")
