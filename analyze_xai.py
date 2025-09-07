#!/usr/bin/env python3
"""
Comprehensive XAI Analysis Script
Analyzes model explanations across multiple samples and generates reports
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import yaml
from typing import Dict, List, Tuple
import os
from tqdm import tqdm

from src.explainable_ai import XAIVisualizer, GradCAM, AttentionVisualizer
from src.data_loader import DataManager
from src.model import create_model
from src.advanced_models import create_advanced_model


class XAIAnalyzer:
    """Comprehensive XAI analysis for medical models"""
    
    def __init__(self, config_path: str = 'configs/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else 
                                 "cuda" if torch.cuda.is_available() else "cpu")
        
        self.results = {}
        
    def load_models(self) -> Dict[str, torch.nn.Module]:
        """Load all available models"""
        models = {}
        
        # Original ViT
        try:
            vit_model = create_model(self.config)
            checkpoint = torch.load('models/best_model.pth', map_location=self.device, weights_only=False)
            vit_model.load_state_dict(checkpoint['model_state_dict'])
            models['Original ViT'] = vit_model
            print("✅ Loaded Original ViT")
        except Exception as e:
            print(f"❌ Could not load Original ViT: {e}")
        
        # Hybrid CNN-ViT
        try:
            hybrid_model = create_advanced_model("hybrid_cnn_vit", self.config)
            if Path('models/best_hybrid_cnn_vit_model.pth').exists():
                checkpoint = torch.load('models/best_hybrid_cnn_vit_model.pth', 
                                      map_location=self.device, weights_only=False)
                hybrid_model.load_state_dict(checkpoint['model_state_dict'])
            models['Hybrid CNN-ViT'] = hybrid_model
            print("✅ Loaded Hybrid CNN-ViT")
        except Exception as e:
            print(f"❌ Could not load Hybrid CNN-ViT: {e}")
        
        # Multi-scale ViT
        try:
            multiscale_model = create_advanced_model("multiscale_vit", self.config)
            if Path('models/best_multiscale_vit_model.pth').exists():
                checkpoint = torch.load('models/best_multiscale_vit_model.pth', 
                                      map_location=self.device, weights_only=False)
                multiscale_model.load_state_dict(checkpoint['model_state_dict'])
            models['Multi-scale ViT'] = multiscale_model
            print("✅ Loaded Multi-scale ViT")
        except Exception as e:
            print(f"❌ Could not load Multi-scale ViT: {e}")
        
        return models
    
    def analyze_model_explanations(self, model, model_name: str, test_loader, 
                                 num_samples: int = 20) -> Dict:
        """Analyze explanations for a specific model"""
        print(f"\n🔍 Analyzing explanations for {model_name}...")
        
        model = model.to(self.device)
        model.eval()
        
        # Initialize XAI components
        xai_viz = XAIVisualizer(model)
        
        explanations_data = []
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(test_loader, desc=f"Processing {model_name}")):
                if i >= num_samples:
                    break
                
                image = images[0].to(self.device)
                true_label = labels[0].item()
                
                # Make prediction
                output = model(image.unsqueeze(0))
                predicted_class = output.argmax(dim=1).item()
                confidence = F.softmax(output, dim=1)[0, predicted_class].item()
                
                # Check if prediction is correct
                is_correct = predicted_class == true_label
                if is_correct:
                    correct_predictions += 1
                total_predictions += 1
                
                # Generate explanations
                try:
                    explanations = xai_viz.create_comprehensive_explanation(
                        image, ['Normal', 'Pneumonia'], predicted_class
                    )
                    
                    # Extract explanation data
                    explanation_data = {
                        'sample_id': i,
                        'true_label': true_label,
                        'predicted_label': predicted_class,
                        'confidence': confidence,
                        'is_correct': is_correct,
                        'has_gradcam': explanations['gradcam'] is not None,
                        'has_attention': explanations['attention_rollout'] is not None,
                        'has_lime': explanations['lime'] is not None
                    }
                    
                    # Add explanation quality metrics if available
                    if explanations['gradcam'] is not None:
                        gradcam = explanations['gradcam']
                        explanation_data['gradcam_entropy'] = self._calculate_entropy(gradcam)
                        explanation_data['gradcam_sparsity'] = self._calculate_sparsity(gradcam)
                    
                    if explanations['attention_rollout'] is not None:
                        attention = explanations['attention_rollout']
                        explanation_data['attention_entropy'] = self._calculate_entropy(attention)
                        explanation_data['attention_sparsity'] = self._calculate_sparsity(attention)
                    
                    explanations_data.append(explanation_data)
                    
                except Exception as e:
                    print(f"Warning: Failed to generate explanations for sample {i}: {e}")
                    continue
        
        # Calculate summary statistics
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        summary = {
            'model_name': model_name,
            'total_samples': total_predictions,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'explanations_data': explanations_data,
            'avg_confidence': np.mean([d['confidence'] for d in explanations_data]) if explanations_data else 0,
            'gradcam_success_rate': np.mean([d['has_gradcam'] for d in explanations_data]) if explanations_data else 0,
            'attention_success_rate': np.mean([d['has_attention'] for d in explanations_data]) if explanations_data else 0,
            'lime_success_rate': np.mean([d['has_lime'] for d in explanations_data]) if explanations_data else 0
        }
        
        return summary
    
    def _calculate_entropy(self, explanation_map: np.ndarray) -> float:
        """Calculate entropy of explanation map"""
        # Normalize to probabilities
        prob_map = explanation_map / explanation_map.sum()
        prob_map = prob_map[prob_map > 0]  # Remove zeros
        entropy = -np.sum(prob_map * np.log2(prob_map))
        return entropy
    
    def _calculate_sparsity(self, explanation_map: np.ndarray, threshold: float = 0.1) -> float:
        """Calculate sparsity of explanation map"""
        normalized_map = explanation_map / explanation_map.max()
        sparsity = np.mean(normalized_map < threshold)
        return sparsity
    
    def compare_explanations(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """Compare explanations across different models"""
        comparison_data = []
        
        for model_name, result in results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': result['accuracy'],
                'Avg Confidence': result['avg_confidence'],
                'GradCAM Success Rate': result['gradcam_success_rate'],
                'Attention Success Rate': result['attention_success_rate'],
                'LIME Success Rate': result['lime_success_rate'],
                'Total Samples': result['total_samples']
            })
        
        return pd.DataFrame(comparison_data)
    
    def plot_explanation_analysis(self, results: Dict[str, Dict], df: pd.DataFrame):
        """Create comprehensive explanation analysis plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('XAI Analysis Across Model Architectures', fontsize=16, fontweight='bold')
        
        models = list(results.keys())
        
        # 1. Accuracy vs Confidence
        ax1 = axes[0, 0]
        accuracies = [results[model]['accuracy'] for model in models]
        confidences = [results[model]['avg_confidence'] for model in models]
        scatter = ax1.scatter(confidences, accuracies, s=100, alpha=0.7)
        for i, model in enumerate(models):
            ax1.annotate(model, (confidences[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax1.set_xlabel('Average Confidence')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy vs Confidence')
        ax1.grid(True, alpha=0.3)
        
        # 2. Explanation Success Rates
        ax2 = axes[0, 1]
        gradcam_rates = [results[model]['gradcam_success_rate'] for model in models]
        attention_rates = [results[model]['attention_success_rate'] for model in models]
        lime_rates = [results[model]['lime_success_rate'] for model in models]
        
        x = np.arange(len(models))
        width = 0.25
        
        ax2.bar(x - width, gradcam_rates, width, label='GradCAM', alpha=0.8)
        ax2.bar(x, attention_rates, width, label='Attention', alpha=0.8)
        ax2.bar(x + width, lime_rates, width, label='LIME', alpha=0.8)
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Success Rate')
        ax2.set_title('Explanation Method Success Rates')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Explanation Quality Metrics
        ax3 = axes[0, 2]
        all_entropies = []
        all_sparsities = []
        model_labels = []
        
        for model_name, result in results.items():
            for data in result['explanations_data']:
                if 'gradcam_entropy' in data:
                    all_entropies.append(data['gradcam_entropy'])
                    all_sparsities.append(data['gradcam_sparsity'])
                    model_labels.append(model_name)
        
        if all_entropies:
            df_quality = pd.DataFrame({
                'Model': model_labels,
                'Entropy': all_entropies,
                'Sparsity': all_sparsities
            })
            
            for model in models:
                model_data = df_quality[df_quality['Model'] == model]
                if not model_data.empty:
                    ax3.scatter(model_data['Entropy'], model_data['Sparsity'], 
                              label=model, alpha=0.6, s=50)
            
            ax3.set_xlabel('Explanation Entropy')
            ax3.set_ylabel('Explanation Sparsity')
            ax3.set_title('Explanation Quality Metrics')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No Quality Metrics\nAvailable', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Explanation Quality Metrics')
        
        # 4. Model Performance Comparison
        ax4 = axes[1, 0]
        metrics = ['Accuracy', 'Avg Confidence', 'GradCAM Success Rate', 'Attention Success Rate']
        metric_values = []
        
        for metric in metrics:
            values = []
            for model in models:
                if metric == 'Accuracy':
                    values.append(results[model]['accuracy'])
                elif metric == 'Avg Confidence':
                    values.append(results[model]['avg_confidence'])
                elif metric == 'GradCAM Success Rate':
                    values.append(results[model]['gradcam_success_rate'])
                elif metric == 'Attention Success Rate':
                    values.append(results[model]['attention_success_rate'])
            metric_values.append(values)
        
        x = np.arange(len(models))
        width = 0.2
        
        for i, (metric, values) in enumerate(zip(metrics, metric_values)):
            ax4.bar(x + i * width, values, width, label=metric, alpha=0.8)
        
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Score')
        ax4.set_title('Model Performance Comparison')
        ax4.set_xticks(x + width * 1.5)
        ax4.set_xticklabels(models, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Sample-wise Analysis
        ax5 = axes[1, 1]
        all_confidences = []
        all_correct = []
        all_models = []
        
        for model_name, result in results.items():
            for data in result['explanations_data']:
                all_confidences.append(data['confidence'])
                all_correct.append(data['is_correct'])
                all_models.append(model_name)
        
        if all_confidences:
            df_samples = pd.DataFrame({
                'Model': all_models,
                'Confidence': all_confidences,
                'Correct': all_correct
            })
            
            # Box plot of confidence by model and correctness
            sns.boxplot(data=df_samples, x='Model', y='Confidence', hue='Correct', ax=ax5)
            ax5.set_title('Confidence Distribution by Model')
            ax5.tick_params(axis='x', rotation=45)
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No Sample Data\nAvailable', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Confidence Distribution')
        
        # 6. Summary Table
        ax6 = axes[1, 2]
        ax6.axis('tight')
        ax6.axis('off')
        
        # Create summary table
        summary_data = df[['Model', 'Accuracy', 'Avg Confidence', 'GradCAM Success Rate']].round(3)
        table = ax6.table(cellText=summary_data.values,
                         colLabels=summary_data.columns,
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax6.set_title('Summary Statistics')
        
        plt.tight_layout()
        plt.savefig('results/xai_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_xai_report(self, results: Dict[str, Dict], df: pd.DataFrame):
        """Generate comprehensive XAI analysis report"""
        report = f"""
# 🔍 Comprehensive XAI Analysis Report

## 📊 Executive Summary

This report analyzes the explainability of different Vision Transformer architectures for medical image classification.

### 🏆 Key Findings:

#### Model Performance:
"""
        
        for model_name, result in results.items():
            report += f"""
- **{model_name}**:
  - Accuracy: {result['accuracy']:.3f}
  - Average Confidence: {result['avg_confidence']:.3f}
  - GradCAM Success Rate: {result['gradcam_success_rate']:.3f}
  - Attention Success Rate: {result['attention_success_rate']:.3f}
  - LIME Success Rate: {result['lime_success_rate']:.3f}
"""
        
        report += f"""

## 📈 Detailed Analysis

### Model Comparison Table:
{df.to_string(index=False)}

### 🔍 Explanation Method Analysis:

#### Grad-CAM:
- **Purpose**: Shows which regions the model focuses on
- **Success Rate**: Varies by model architecture
- **Best Performer**: {df.loc[df['GradCAM Success Rate'].idxmax(), 'Model']}

#### Attention Visualization:
- **Purpose**: Reveals transformer attention patterns
- **Success Rate**: Generally high for ViT-based models
- **Best Performer**: {df.loc[df['Attention Success Rate'].idxmax(), 'Model']}

#### LIME:
- **Purpose**: Identifies important features
- **Success Rate**: May vary due to computational complexity
- **Best Performer**: {df.loc[df['LIME Success Rate'].idxmax(), 'Model']}

## 🎯 Recommendations:

1. **For Clinical Use**: Choose models with high explanation success rates
2. **For Research**: Use models with diverse explanation capabilities
3. **For Deployment**: Consider computational requirements of explanation methods

## 📋 Technical Details:

- **Total Samples Analyzed**: {sum(result['total_samples'] for result in results.values())}
- **Analysis Methods**: Grad-CAM, Attention Rollout, LIME
- **Quality Metrics**: Entropy, Sparsity, Success Rates

---
*Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open('results/xai_analysis_report.md', 'w') as f:
            f.write(report)
        
        print("📄 XAI analysis report saved to results/xai_analysis_report.md")


def main():
    """Main XAI analysis function"""
    print("🔍 Starting Comprehensive XAI Analysis...")
    
    # Initialize analyzer
    analyzer = XAIAnalyzer()
    
    # Load data
    print("📊 Loading test data...")
    data_manager = DataManager(analyzer.config)
    _, _, test_loader, _ = data_manager.create_dataloaders()
    
    # Load models
    print("🤖 Loading models...")
    models = analyzer.load_models()
    
    if not models:
        print("❌ No models available for analysis!")
        return
    
    # Analyze each model
    results = {}
    for model_name, model in models.items():
        result = analyzer.analyze_model_explanations(model, model_name, test_loader, num_samples=20)
        results[model_name] = result
    
    # Create comparison
    print("📊 Creating comparison analysis...")
    df = analyzer.compare_explanations(results)
    
    # Generate visualizations
    print("📈 Generating analysis plots...")
    analyzer.plot_explanation_analysis(results, df)
    
    # Generate report
    print("📄 Generating comprehensive report...")
    analyzer.generate_xai_report(results, df)
    
    print("✅ XAI analysis completed!")
    print(f"📁 Results saved in results/ directory")


if __name__ == "__main__":
    main()
