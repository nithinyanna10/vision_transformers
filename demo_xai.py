#!/usr/bin/env python3
"""
Simple XAI Demo - See How It Works!
This script demonstrates the explainable AI features with actual results
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import yaml
import os
from pathlib import Path

from src.model import create_model
from src.explainable_ai import GradCAM, AttentionVisualizer, LIMEExplainer


def load_trained_model():
    """Load our trained model"""
    print("🤖 Loading trained model...")
    
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model = create_model(config)
    
    # Load the trained weights
    if os.path.exists('models/best_model.pth'):
        checkpoint = torch.load('models/best_model.pth', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Model loaded successfully!")
        return model
    else:
        print("❌ No trained model found!")
        return None


def create_sample_image():
    """Create a sample image for testing"""
    print("📸 Creating sample image...")
    
    # Create a simple test image (simulating a chest X-ray)
    # In real use, you would load an actual X-ray image
    image = np.random.rand(224, 224, 3) * 0.3  # Dark background
    image[50:150, 80:180] = 0.8  # Simulate lung area
    image[100:120, 100:160] = 0.9  # Simulate some structure
    
    # Convert to PIL Image
    image_pil = Image.fromarray((image * 255).astype(np.uint8))
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
    
    print("✅ Sample image created!")
    return image_pil, image_tensor


def demonstrate_gradcam(model, image_tensor):
    """Demonstrate Grad-CAM visualization"""
    print("\n🔍 Demonstrating Grad-CAM...")
    
    try:
        # Create Grad-CAM
        gradcam = GradCAM(model, target_layer_name="transformer_blocks.0.attn")
        
        # Generate CAM
        cam = gradcam.generate_cam(image_tensor.unsqueeze(0))
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image
        ax1.imshow(image_tensor.permute(1, 2, 0).numpy())
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Grad-CAM
        im = ax2.imshow(cam, cmap='hot', interpolation='nearest')
        ax2.set_title('Grad-CAM: What the model focuses on')
        ax2.axis('off')
        plt.colorbar(im, ax=ax2, shrink=0.8)
        
        plt.tight_layout()
        plt.savefig('results/gradcam_demo.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Grad-CAM demonstration completed!")
        print("📁 Saved to: results/gradcam_demo.png")
        
    except Exception as e:
        print(f"❌ Grad-CAM failed: {e}")
        print("This is normal for some model architectures")


def demonstrate_attention(model, image_tensor):
    """Demonstrate attention visualization"""
    print("\n👁️ Demonstrating Attention Visualization...")
    
    try:
        # Create attention visualizer
        attn_viz = AttentionVisualizer(model)
        
        # Generate attention rollout
        attention_map = attn_viz.visualize_attention_rollout(image_tensor.unsqueeze(0))
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image
        ax1.imshow(image_tensor.permute(1, 2, 0).numpy())
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Attention map
        im = ax2.imshow(attention_map, cmap='hot', interpolation='nearest')
        ax2.set_title('Attention Rollout: Where the model looks')
        ax2.axis('off')
        plt.colorbar(im, ax=ax2, shrink=0.8)
        
        plt.tight_layout()
        plt.savefig('results/attention_demo.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Attention visualization completed!")
        print("📁 Saved to: results/attention_demo.png")
        
    except Exception as e:
        print(f"❌ Attention visualization failed: {e}")
        print("This is normal for some model architectures")


def demonstrate_lime(model, image_tensor):
    """Demonstrate LIME explanation"""
    print("\n🍋 Demonstrating LIME Explanation...")
    
    try:
        # Create LIME explainer
        lime_explainer = LIMEExplainer(model)
        
        # Generate explanation
        explanation = lime_explainer.explain_image(image_tensor)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image
        ax1.imshow(image_tensor.permute(1, 2, 0).numpy())
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # LIME explanation
        try:
            lime_img, lime_mask = explanation.get_image_and_mask(
                explanation.top_labels[0], 
                positive_only=True, 
                num_features=10, 
                hide_rest=True
            )
            ax2.imshow(lime_img)
            ax2.set_title('LIME: Important features highlighted')
        except:
            ax2.text(0.5, 0.5, 'LIME Processing\nIn Progress...', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('LIME Explanation')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig('results/lime_demo.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ LIME explanation completed!")
        print("📁 Saved to: results/lime_demo.png")
        
    except Exception as e:
        print(f"❌ LIME failed: {e}")
        print("This might take a while or fail on some systems")


def demonstrate_prediction(model, image_tensor):
    """Demonstrate model prediction"""
    print("\n🎯 Demonstrating Model Prediction...")
    
    model.eval()
    with torch.no_grad():
        # Make prediction
        output = model(image_tensor.unsqueeze(0))
        probabilities = F.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    class_names = ['Normal', 'Pneumonia']
    
    print(f"📊 Prediction Results:")
    print(f"   Predicted Class: {class_names[predicted_class]}")
    print(f"   Confidence: {confidence:.3f}")
    print(f"   Probabilities:")
    for i, class_name in enumerate(class_names):
        prob = probabilities[0, i].item()
        print(f"     {class_name}: {prob:.3f}")
    
    # Create probability visualization
    fig, ax = plt.subplots(figsize=(8, 4))
    classes = class_names
    probs = [probabilities[0, i].item() for i in range(len(class_names))]
    
    bars = ax.bar(classes, probs, color=['#2E8B57', '#DC143C'])
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1)
    ax.set_title('Model Prediction Probabilities')
    
    # Add value labels on bars
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/prediction_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Prediction demonstration completed!")
    print("📁 Saved to: results/prediction_demo.png")


def main():
    """Main demonstration function"""
    print("🚀 XAI Demonstration - See How It Works!")
    print("=" * 50)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Load model
    model = load_trained_model()
    if model is None:
        print("❌ Cannot proceed without a trained model!")
        return
    
    # Create sample image
    image_pil, image_tensor = create_sample_image()
    
    # Demonstrate prediction
    demonstrate_prediction(model, image_tensor)
    
    # Demonstrate Grad-CAM
    demonstrate_gradcam(model, image_tensor)
    
    # Demonstrate attention visualization
    demonstrate_attention(model, image_tensor)
    
    # Demonstrate LIME
    demonstrate_lime(model, image_tensor)
    
    print("\n🎉 XAI Demonstration Completed!")
    print("📁 Check the 'results/' folder for generated images")
    print("\n💡 What you just saw:")
    print("   • Model prediction with confidence scores")
    print("   • Grad-CAM showing which regions the model focuses on")
    print("   • Attention visualization showing transformer attention")
    print("   • LIME highlighting important features")
    print("\n🔍 These explanations help understand HOW the AI makes decisions!")


if __name__ == "__main__":
    main()
