#!/usr/bin/env python3
"""
Interactive XAI Demo Application
Streamlit app for exploring model explanations
"""

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import yaml
import os
from pathlib import Path

from src.explainable_ai import XAIVisualizer, GradCAM, AttentionVisualizer, LIMEExplainer
from src.model import create_model
from src.advanced_models import create_advanced_model
import albumentations as A
from albumentations.pytorch import ToTensorV2


def load_model(model_type: str = "original"):
    """Load the specified model"""
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    if model_type == "original":
        model = create_model(config)
        model_path = 'models/best_model.pth'
    elif model_type == "hybrid":
        model = create_advanced_model("hybrid_cnn_vit", config)
        model_path = 'models/best_hybrid_cnn_vit_model.pth'
    elif model_type == "multiscale":
        model = create_advanced_model("multiscale_vit", config)
        model_path = 'models/best_multiscale_vit_model.pth'
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights if available
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        st.success(f"✅ Loaded {model_type} model from {model_path}")
    else:
        st.warning(f"⚠️ Model weights not found at {model_path}. Using untrained model.")
    
    return model


def preprocess_image(image, img_size=224):
    """Preprocess uploaded image"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize((img_size, img_size))
    
    # Convert to numpy
    image_np = np.array(image)
    
    # Define transform
    transform = A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    # Apply transform
    transformed = transform(image=image_np)
    image_tensor = transformed['image']
    
    return image_tensor, image_np


def predict_image(model, image_tensor, class_names):
    """Make prediction on image"""
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
        probabilities = F.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': {
            class_names[i]: probabilities[0, i].item() 
            for i in range(len(class_names))
        }
    }


def main():
    st.set_page_config(
        page_title="Medical ViT XAI Demo",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Medical Vision Transformer - Explainable AI Demo")
    st.markdown("Explore how our AI model makes decisions with advanced explainability techniques")
    
    # Sidebar for model selection
    st.sidebar.header("🎛️ Model Selection")
    model_type = st.sidebar.selectbox(
        "Choose Model Architecture:",
        ["original", "hybrid", "multiscale"],
        format_func=lambda x: {
            "original": "Original ViT",
            "hybrid": "Hybrid CNN-ViT", 
            "multiscale": "Multi-scale ViT"
        }[x]
    )
    
    # Load model
    with st.spinner(f"Loading {model_type} model..."):
        try:
            model = load_model(model_type)
            st.sidebar.success(f"✅ {model_type.title()} model loaded!")
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            return
    
    # Class names
    class_names = ['Normal', 'Pneumonia']
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Upload Chest X-ray")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a chest X-ray image for analysis"
        )
        
        if uploaded_file is not None:
            # Load and preprocess image
            image = Image.open(uploaded_file)
            image_tensor, image_np = preprocess_image(image)
            
            # Display original image
            st.subheader("Original Image")
            st.image(image, caption="Uploaded Chest X-ray", use_column_width=True)
            
            # Make prediction
            with st.spinner("Making prediction..."):
                prediction = predict_image(model, image_tensor, class_names)
            
            # Display prediction results
            st.subheader("🎯 Prediction Results")
            
            predicted_class = prediction['predicted_class']
            confidence = prediction['confidence']
            
            if predicted_class == 0:
                st.success(f"**Normal** - No signs of pneumonia detected")
                st.success(f"Confidence: {confidence:.1%}")
            else:
                st.error(f"**Pneumonia** - Signs of pneumonia detected")
                st.error(f"Confidence: {confidence:.1%}")
            
            # Probability breakdown
            st.subheader("📊 Probability Breakdown")
            prob_data = prediction['probabilities']
            
            fig, ax = plt.subplots(figsize=(8, 4))
            classes = list(prob_data.keys())
            probs = list(prob_data.values())
            
            bars = ax.bar(classes, probs, color=['#2E8B57', '#DC143C'])
            ax.set_ylabel('Probability')
            ax.set_ylim(0, 1)
            ax.set_title('Class Probabilities')
            
            # Add value labels on bars
            for bar, prob in zip(bars, probs):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
            
            st.pyplot(fig)
    
    with col2:
        st.header("🔍 Explainability Analysis")
        
        if uploaded_file is not None:
            # XAI analysis options
            st.subheader("🎛️ Analysis Options")
            
            col_a, col_b = st.columns(2)
            with col_a:
                show_gradcam = st.checkbox("Grad-CAM", value=True)
                show_attention = st.checkbox("Attention Visualization", value=True)
            with col_b:
                show_lime = st.checkbox("LIME Explanation", value=True)
                show_combined = st.checkbox("Combined View", value=True)
            
            # Generate explanations
            if st.button("🔍 Generate Explanations", type="primary"):
                with st.spinner("Generating explanations..."):
                    try:
                        # Initialize XAI visualizer
                        xai_viz = XAIVisualizer(model)
                        
                        # Generate comprehensive explanations
                        explanations = xai_viz.create_comprehensive_explanation(
                            image_tensor, class_names
                        )
                        
                        # Display explanations
                        st.subheader("🧠 Model Explanations")
                        
                        # Create explanation plots
                        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                        fig.suptitle('XAI Explanations', fontsize=16, fontweight='bold')
                        
                        # Grad-CAM
                        if show_gradcam and explanations['gradcam'] is not None:
                            ax = axes[0, 0]
                            ax.imshow(explanations['gradcam'])
                            ax.set_title('Grad-CAM: What the model focuses on')
                            ax.axis('off')
                        else:
                            axes[0, 0].text(0.5, 0.5, 'Grad-CAM\nNot Available', 
                                          ha='center', va='center', transform=axes[0, 0].transAxes)
                            axes[0, 0].set_title('Grad-CAM')
                            axes[0, 0].axis('off')
                        
                        # Attention Rollout
                        if show_attention and explanations['attention_rollout'] is not None:
                            ax = axes[0, 1]
                            im = ax.imshow(explanations['attention_rollout'], cmap='hot', interpolation='nearest')
                            ax.set_title('Attention Rollout: Transformer attention')
                            ax.axis('off')
                            plt.colorbar(im, ax=ax, shrink=0.8)
                        else:
                            axes[0, 1].text(0.5, 0.5, 'Attention\nNot Available', 
                                          ha='center', va='center', transform=axes[0, 1].transAxes)
                            axes[0, 1].set_title('Attention Rollout')
                            axes[0, 1].axis('off')
                        
                        # LIME Explanation
                        if show_lime and explanations['lime'] is not None:
                            ax = axes[1, 0]
                            try:
                                lime_img, lime_mask = explanations['lime'].get_image_and_mask(
                                    explanations['lime'].top_labels[0], 
                                    positive_only=True, 
                                    num_features=10, 
                                    hide_rest=True
                                )
                                ax.imshow(lime_img)
                                ax.set_title('LIME: Feature importance')
                            except:
                                ax.text(0.5, 0.5, 'LIME\nProcessing Error', 
                                      ha='center', va='center', transform=axes[1, 0].transAxes)
                                ax.set_title('LIME Explanation')
                            ax.axis('off')
                        else:
                            axes[1, 0].text(0.5, 0.5, 'LIME\nNot Available', 
                                          ha='center', va='center', transform=axes[1, 0].transAxes)
                            axes[1, 0].set_title('LIME Explanation')
                            axes[1, 0].axis('off')
                        
                        # Combined Visualization
                        if show_combined:
                            ax = axes[1, 1]
                            if (explanations['gradcam'] is not None and 
                                explanations['attention_rollout'] is not None):
                                # Combine Grad-CAM and attention
                                combined = 0.5 * explanations['gradcam'] + 0.5 * explanations['attention_rollout']
                                ax.imshow(combined, cmap='viridis')
                                ax.set_title('Combined: Grad-CAM + Attention')
                            else:
                                ax.text(0.5, 0.5, 'Combined\nNot Available', 
                                      ha='center', va='center', transform=axes[1, 1].transAxes)
                                ax.set_title('Combined Visualization')
                            ax.axis('off')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Explanation insights
                        st.subheader("💡 Explanation Insights")
                        
                        insights = [
                            "**Grad-CAM**: Shows which regions of the image the model focuses on for its decision",
                            "**Attention Rollout**: Reveals how the Vision Transformer processes different parts of the image",
                            "**LIME**: Identifies the most important features that influence the prediction",
                            "**Combined View**: Merges multiple explanation methods for comprehensive understanding"
                        ]
                        
                        for insight in insights:
                            st.markdown(f"• {insight}")
                        
                        st.success("✅ Explanations generated successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Failed to generate explanations: {e}")
                        st.error("This might be due to model architecture compatibility or missing dependencies.")
        
        else:
            st.info("👆 Upload an image to see explainability analysis")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 🔬 About This Demo
    
    This interactive demo showcases advanced explainable AI techniques for medical image classification:
    
    - **Grad-CAM**: Gradient-weighted Class Activation Mapping shows which regions the model focuses on
    - **Attention Visualization**: Reveals how Vision Transformers process different image regions
    - **LIME**: Local Interpretable Model-agnostic Explanations identify important features
    - **Combined Analysis**: Merges multiple explanation methods for comprehensive insights
    
    ### 🎯 Use Cases
    
    - **Medical Professionals**: Understand AI decision-making for clinical validation
    - **Researchers**: Analyze model behavior and identify potential biases
    - **Students**: Learn about explainable AI in medical imaging
    - **Developers**: Debug and improve model performance
    
    ### ⚠️ Disclaimer
    
    This tool is for educational and research purposes only. It should not be used for clinical diagnosis.
    """)
    
    # Model information
    with st.expander("📋 Model Information"):
        st.markdown(f"""
        **Selected Model**: {model_type.title()}
        
        **Architecture Details**:
        - Original ViT: Pure Vision Transformer with 12 layers
        - Hybrid CNN-ViT: ResNet50 backbone + ViT with cross-attention
        - Multi-scale ViT: Parallel processing with different patch sizes
        
        **Parameters**: {sum(p.numel() for p in model.parameters()):,}
        
        **Training**: Trained on Chest X-ray Pneumonia dataset
        """)


if __name__ == "__main__":
    main()
