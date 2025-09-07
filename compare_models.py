#!/usr/bin/env python3
"""
Model Comparison Script for Advanced Medical ViT Architectures
Compares different model architectures and creates comprehensive reports
"""

import torch
import torch.nn as nn
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import time
from typing import Dict, List, Tuple

from src.advanced_models import create_advanced_model, load_pretrained_models, EnsembleModel
from src.data_loader import DataManager
from src.evaluator import evaluate_model
from src.model import create_model


class ModelComparator:
    """Compare different model architectures"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else 
                                 "cuda" if torch.cuda.is_available() else "cpu")
        
        self.results = {}
        
    def load_models(self) -> Dict[str, nn.Module]:
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
        
        # Ensemble Model
        try:
            ensemble_model = create_advanced_model("ensemble", self.config)
            models['Ensemble'] = ensemble_model
            print("✅ Loaded Ensemble Model")
        except Exception as e:
            print(f"❌ Could not load Ensemble Model: {e}")
        
        return models
    
    def evaluate_models(self, models: Dict[str, nn.Module], test_loader) -> Dict[str, Dict]:
        """Evaluate all models on test set"""
        results = {}
        
        for name, model in models.items():
            print(f"\n🧪 Evaluating {name}...")
            model = model.to(self.device)
            model.eval()
            
            # Measure inference time
            start_time = time.time()
            metrics = evaluate_model(model, test_loader, ['Normal', 'Pneumonia'], self.device)
            inference_time = time.time() - start_time
            
            # Add model info
            metrics['inference_time'] = inference_time
            metrics['model_size_mb'] = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            metrics['num_parameters'] = sum(p.numel() for p in model.parameters())
            
            results[name] = metrics
            
            print(f"✅ {name} - Accuracy: {metrics['accuracy']:.4f}, Time: {inference_time:.2f}s")
        
        return results
    
    def create_comparison_report(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """Create comprehensive comparison report"""
        # Convert to DataFrame
        df = pd.DataFrame(results).T
        
        # Round numerical values
        numeric_cols = ['accuracy', 'precision', 'recall', 'f1_score', 'auc', 'inference_time']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].round(4)
        
        # Add ranking
        df['accuracy_rank'] = df['accuracy'].rank(ascending=False, method='dense')
        df['speed_rank'] = df['inference_time'].rank(ascending=True, method='dense')
        df['efficiency_rank'] = (df['accuracy'] / df['inference_time']).rank(ascending=False, method='dense')
        
        return df
    
    def plot_comparison_charts(self, results: Dict[str, Dict], df: pd.DataFrame):
        """Create comprehensive comparison charts"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Advanced Model Architecture Comparison', fontsize=16, fontweight='bold')
        
        # 1. Accuracy Comparison
        ax1 = axes[0, 0]
        models = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in models]
        bars = ax1.bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax1.set_title('Test Accuracy Comparison', fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0.8, 1.0)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{accuracies[i]:.4f}', ha='center', va='bottom', fontweight='bold')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # 2. Performance Metrics Radar Chart
        ax2 = axes[0, 1]
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, (model, result) in enumerate(results.items()):
            values = [result[metric] for metric in metrics]
            values += values[:1]  # Complete the circle
            ax2.plot(angles, values, 'o-', linewidth=2, label=model)
            ax2.fill(angles, values, alpha=0.25)
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax2.set_ylim(0.8, 1.0)
        ax2.set_title('Performance Metrics Radar', fontweight='bold')
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax2.grid(True)
        
        # 3. Speed vs Accuracy
        ax3 = axes[0, 2]
        times = [results[model]['inference_time'] for model in models]
        ax3.scatter(times, accuracies, s=100, alpha=0.7)
        for i, model in enumerate(models):
            ax3.annotate(model, (times[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax3.set_xlabel('Inference Time (seconds)')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Speed vs Accuracy Trade-off', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Model Size Comparison
        ax4 = axes[1, 0]
        sizes = [results[model]['model_size_mb'] for model in models]
        bars = ax4.bar(models, sizes, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax4.set_title('Model Size Comparison', fontweight='bold')
        ax4.set_ylabel('Size (MB)')
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(sizes)*0.01,
                    f'{sizes[i]:.1f}MB', ha='center', va='bottom', fontweight='bold')
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        # 5. Efficiency Score
        ax5 = axes[1, 1]
        efficiency = [acc / time for acc, time in zip(accuracies, times)]
        bars = ax5.bar(models, efficiency, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax5.set_title('Efficiency Score (Accuracy/Time)', fontweight='bold')
        ax5.set_ylabel('Efficiency')
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + max(efficiency)*0.01,
                    f'{efficiency[i]:.2f}', ha='center', va='bottom', fontweight='bold')
        plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
        
        # 6. Ranking Summary
        ax6 = axes[1, 2]
        ranking_data = df[['accuracy_rank', 'speed_rank', 'efficiency_rank']].T
        im = ax6.imshow(ranking_data.values, cmap='RdYlGn_r', aspect='auto')
        ax6.set_xticks(range(len(models)))
        ax6.set_xticklabels(models, rotation=45, ha='right')
        ax6.set_yticks(range(3))
        ax6.set_yticklabels(['Accuracy Rank', 'Speed Rank', 'Efficiency Rank'])
        ax6.set_title('Model Rankings', fontweight='bold')
        
        # Add text annotations
        for i in range(3):
            for j in range(len(models)):
                text = ax6.text(j, i, f'{ranking_data.iloc[i, j]:.0f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax6, shrink=0.8)
        
        plt.tight_layout()
        plt.savefig('models/advanced_models_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, df: pd.DataFrame):
        """Generate comprehensive text report"""
        report = f"""
# 🏥 Advanced Medical ViT Models Comparison Report

## 📊 Executive Summary

This report compares different advanced architectures for medical image classification:

### 🏆 Best Performing Models:
- **Highest Accuracy**: {df.loc[df['accuracy_rank'] == 1.0].index[0]} ({df.loc[df['accuracy_rank'] == 1.0, 'accuracy'].iloc[0]:.4f})
- **Fastest Inference**: {df.loc[df['speed_rank'] == 1.0].index[0]} ({df.loc[df['speed_rank'] == 1.0, 'inference_time'].iloc[0]:.2f}s)
- **Most Efficient**: {df.loc[df['efficiency_rank'] == 1.0].index[0]} (Efficiency: {df.loc[df['efficiency_rank'] == 1.0, 'efficiency_rank'].iloc[0]:.2f})

## 📈 Detailed Results

{df.to_string()}

## 🔍 Key Insights

1. **Architecture Trade-offs**: Different architectures show different strengths
2. **Performance vs Speed**: There's a clear trade-off between accuracy and inference speed
3. **Model Complexity**: Larger models generally perform better but are slower
4. **Efficiency**: Some models achieve good accuracy with reasonable speed

## 🎯 Recommendations

- **For Production**: Choose based on your specific requirements (accuracy vs speed)
- **For Research**: Use the highest accuracy model for further development
- **For Deployment**: Consider the efficiency score for balanced performance

---
*Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open('models/advanced_models_report.md', 'w') as f:
            f.write(report)
        
        print("📄 Comprehensive report saved to models/advanced_models_report.md")


def main():
    """Main comparison function"""
    print("🚀 Starting Advanced Model Comparison...")
    
    # Initialize comparator
    comparator = ModelComparator('configs/config.yaml')
    
    # Load data
    print("📊 Loading test data...")
    data_manager = DataManager(comparator.config)
    _, _, test_loader, _ = data_manager.create_dataloaders()
    
    # Load models
    print("🤖 Loading models...")
    models = comparator.load_models()
    
    if not models:
        print("❌ No models available for comparison!")
        return
    
    # Evaluate models
    print("🧪 Evaluating models...")
    results = comparator.evaluate_models(models, test_loader)
    
    # Create comparison report
    print("📊 Creating comparison report...")
    df = comparator.create_comparison_report(results)
    
    # Generate visualizations
    print("📈 Generating comparison charts...")
    comparator.plot_comparison_charts(results, df)
    
    # Generate text report
    print("📄 Generating comprehensive report...")
    comparator.generate_report(df)
    
    print("✅ Model comparison completed!")
    print(f"📁 Results saved in models/ directory")


if __name__ == "__main__":
    main()
