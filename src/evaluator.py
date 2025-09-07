"""
Advanced Evaluation and Visualization Tools for Medical ViT
Includes attention visualization, Grad-CAM, confusion matrices, and clinical metrics
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    precision_recall_fscore_support, roc_auc_score
)
from sklearn.metrics import cohen_kappa_score
import cv2
from PIL import Image
import os
from tqdm import tqdm
import yaml


class AttentionVisualizer:
    """Visualize attention maps from Vision Transformer"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def get_attention_maps(self, image, layer_idx=-1):
        """Get attention maps from specified layer"""
        self.model.eval()
        
        with torch.no_grad():
            # Get attention weights
            attention_weights = self.model.get_attention_maps(image, layer_idx)
            
            if attention_weights is not None:
                # Average across heads
                attention_weights = attention_weights.mean(dim=1)  # B, num_patches
                
                # Reshape to spatial dimensions
                patch_size = int(np.sqrt(attention_weights.shape[1]))
                attention_weights = attention_weights.reshape(-1, patch_size, patch_size)
                
                return attention_weights.cpu().numpy()
        
        return None
    
    def visualize_attention(self, image, attention_maps, save_path=None):
        """Visualize attention maps overlaid on original image"""
        if attention_maps is None:
            return None
            
        # Convert tensor to numpy
        if isinstance(image, torch.Tensor):
            image = image.squeeze().cpu().numpy()
            if image.shape[0] == 3:  # CHW format
                image = np.transpose(image, (1, 2, 0))
        
        # Denormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0, 1)
        
        # Resize attention map to match image size
        attention_map = attention_maps[0]  # Take first sample
        attention_map = cv2.resize(attention_map, (image.shape[1], image.shape[0]))
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Attention map
        im1 = axes[1].imshow(attention_map, cmap='jet')
        axes[1].set_title('Attention Map')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1])
        
        # Overlay
        axes[2].imshow(image)
        axes[2].imshow(attention_map, cmap='jet', alpha=0.5)
        axes[2].set_title('Attention Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig


class GradCAM:
    """Gradient-weighted Class Activation Mapping for ViT"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.gradients = None
        self.activations = None
        
    def save_gradient(self, grad):
        """Save gradients for Grad-CAM"""
        self.gradients = grad
    
    def get_gradcam(self, image, target_class=None):
        """Generate Grad-CAM visualization"""
        self.model.eval()
        
        # Forward pass
        image = image.to(self.device)
        image.requires_grad_()
        
        # Get activations from the last transformer block
        x = self.model.patch_embed(image)
        cls_tokens = self.model.cls_token.expand(image.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.model.pos_embed(x)
        
        # Apply transformer blocks
        for i, block in enumerate(self.model.blocks):
            if i == len(self.model.blocks) - 1:  # Last block
                # Register hook for gradients
                x.register_hook(self.save_gradient)
                self.activations = x
            x = block(x)
        
        # Final classification
        x = self.model.norm(x)
        cls_output = x[:, 0]
        logits = self.model.head(cls_output)
        
        # Backward pass
        if target_class is None:
            target_class = logits.argmax(dim=1)
        
        self.model.zero_grad()
        logits[0, target_class].backward()
        
        # Generate Grad-CAM
        gradients = self.gradients[0, 1:, :]  # Remove cls token
        activations = self.activations[0, 1:, :]  # Remove cls token
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=1)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[0])
        for i, w in enumerate(weights):
            cam += w * activations[:, i]
        
        # Reshape to spatial dimensions
        patch_size = int(np.sqrt(cam.shape[0]))
        cam = cam.reshape(patch_size, patch_size)
        
        # Normalize
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam.cpu().numpy()


class MedicalEvaluator:
    """Comprehensive evaluation for medical image classification"""
    
    def __init__(self, model, device, class_names):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.attention_visualizer = AttentionVisualizer(model, device)
        self.gradcam = GradCAM(model, device)
        
    def evaluate_model(self, test_loader, save_dir=None):
        """Comprehensive model evaluation"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_attention_maps = []
        
        print("Evaluating model...")
        
        with torch.no_grad():
            for data, target in tqdm(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Get predictions
                output = self.model(data)
                probabilities = torch.softmax(output, dim=1)
                predictions = output.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Get attention maps for visualization
                attention_maps = self.attention_visualizer.get_attention_maps(data)
                if attention_maps is not None:
                    all_attention_maps.extend(attention_maps)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_targets, all_predictions, all_probabilities)
        
        # Generate visualizations
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self.plot_confusion_matrix(all_targets, all_predictions, 
                                     os.path.join(save_dir, 'confusion_matrix.png'))
            self.plot_roc_curves(all_targets, all_probabilities, 
                               os.path.join(save_dir, 'roc_curves.png'))
            self.plot_precision_recall_curves(all_targets, all_probabilities,
                                            os.path.join(save_dir, 'pr_curves.png'))
        
        return metrics, all_predictions, all_targets, all_probabilities
    
    def calculate_metrics(self, targets, predictions, probabilities):
        """Calculate comprehensive evaluation metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = np.mean(np.array(targets) == np.array(predictions))
        
        # Classification report
        report = classification_report(targets, predictions, 
                                     target_names=self.class_names, 
                                     output_dict=True)
        metrics['classification_report'] = report
        
        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted'
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1
        
        # AUC
        try:
            if len(np.unique(targets)) > 1:  # Check if we have both classes
                auc_score = roc_auc_score(targets, np.array(probabilities)[:, 1])
                metrics['auc'] = auc_score
            else:
                metrics['auc'] = 0.5  # Random performance if only one class
        except:
            metrics['auc'] = 0.0
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(targets, predictions)
        metrics['cohen_kappa'] = kappa
        
        # Average Precision
        try:
            avg_precision = average_precision_score(targets, np.array(probabilities)[:, 1])
            metrics['average_precision'] = avg_precision
        except:
            metrics['average_precision'] = 0.0
        
        return metrics
    
    def plot_confusion_matrix(self, targets, predictions, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, targets, probabilities, save_path=None):
        """Plot ROC curves"""
        fpr, tpr, _ = roc_curve(targets, np.array(probabilities)[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curves(self, targets, probabilities, save_path=None):
        """Plot Precision-Recall curves"""
        precision, recall, _ = precision_recall_curve(targets, np.array(probabilities)[:, 1])
        avg_precision = average_precision_score(targets, np.array(probabilities)[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AP = {avg_precision:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_sample_predictions(self, test_loader, num_samples=5, save_dir=None):
        """Visualize sample predictions with attention maps"""
        self.model.eval()
        
        samples_visualized = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                if samples_visualized >= num_samples:
                    break
                    
                data, target = data.to(self.device), target.to(self.device)
                
                # Get predictions
                output = self.model(data)
                probabilities = torch.softmax(output, dim=1)
                predictions = output.argmax(dim=1)
                
                # Get attention maps
                attention_maps = self.attention_visualizer.get_attention_maps(data)
                
                # Visualize each sample in the batch
                for i in range(min(data.shape[0], num_samples - samples_visualized)):
                    # Create visualization
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Original image
                    image = data[i].cpu().numpy()
                    if image.shape[0] == 3:
                        image = np.transpose(image, (1, 2, 0))
                    
                    # Denormalize
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    image = image * std + mean
                    image = np.clip(image, 0, 1)
                    
                    axes[0].imshow(image)
                    axes[0].set_title(f'Original\nTrue: {self.class_names[target[i]]}')
                    axes[0].axis('off')
                    
                    # Attention map
                    if attention_maps is not None:
                        att_map = attention_maps[i]
                        att_map = cv2.resize(att_map, (image.shape[1], image.shape[0]))
                        
                        im = axes[1].imshow(att_map, cmap='jet')
                        axes[1].set_title('Attention Map')
                        axes[1].axis('off')
                        plt.colorbar(im, ax=axes[1])
                        
                        # Overlay
                        axes[2].imshow(image)
                        axes[2].imshow(att_map, cmap='jet', alpha=0.5)
                        axes[2].set_title(f'Overlay\nPred: {self.class_names[predictions[i]]}\nConf: {probabilities[i, predictions[i]]:.3f}')
                        axes[2].axis('off')
                    
                    plt.tight_layout()
                    
                    if save_dir:
                        plt.savefig(os.path.join(save_dir, f'sample_{samples_visualized}.png'),
                                  dpi=300, bbox_inches='tight')
                    plt.show()
                    
                    samples_visualized += 1


def evaluate_trained_model(model_path, config, test_loader, class_names):
    """Evaluate a trained model"""
    
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Load model
    from src.model import create_model
    model = create_model(config).to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create evaluator
    evaluator = MedicalEvaluator(model, device, class_names)
    
    # Evaluate
    results_dir = 'results'
    metrics, predictions, targets, probabilities = evaluator.evaluate_model(
        test_loader, results_dir
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
    print(f"Average Precision: {metrics['average_precision']:.4f}")
    
    # Visualize sample predictions
    evaluator.visualize_sample_predictions(test_loader, num_samples=5, save_dir=results_dir)
    
    return metrics


if __name__ == "__main__":
    # This would be called after training
    pass
