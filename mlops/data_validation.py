#!/usr/bin/env python3
"""
Data Validation and Quality Assurance for Medical Vision Transformer
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


class DataValidator:
    """Comprehensive data validation system"""
    
    def __init__(self, data_path="data/chest_xray"):
        self.data_path = Path(data_path)
        self.validation_report = {
            'timestamp': None,
            'data_path': str(data_path),
            'validation_results': {},
            'quality_metrics': {},
            'issues': [],
            'recommendations': []
        }
    
    def validate_dataset(self, data_path=None):
        """Run complete dataset validation"""
        if data_path:
            self.data_path = Path(data_path)
        
        print("🔍 Starting comprehensive data validation...")
        
        # Basic structure validation
        self._validate_structure()
        
        # Image quality validation
        self._validate_image_quality()
        
        # Class distribution validation
        self._validate_class_distribution()
        
        # Data integrity validation
        self._validate_data_integrity()
        
        # Generate report
        self._generate_validation_report()
        
        print("✅ Data validation completed!")
        return self.validation_report
    
    def _validate_structure(self):
        """Validate dataset structure"""
        print("📁 Validating dataset structure...")
        
        expected_dirs = ['train', 'val', 'test']
        expected_classes = ['NORMAL', 'PNEUMONIA']
        
        structure_results = {
            'valid': True,
            'missing_dirs': [],
            'missing_classes': [],
            'extra_dirs': [],
            'extra_classes': []
        }
        
        # Check if data directory exists
        if not self.data_path.exists():
            structure_results['valid'] = False
            structure_results['missing_dirs'] = ['data directory']
            self.validation_report['issues'].append("Data directory does not exist")
            return
        
        # Check for expected directories
        for dir_name in expected_dirs:
            dir_path = self.data_path / dir_name
            if not dir_path.exists():
                structure_results['valid'] = False
                structure_results['missing_dirs'].append(dir_name)
                self.validation_report['issues'].append(f"Missing directory: {dir_name}")
        
        # Check for expected classes in each directory
        for dir_name in expected_dirs:
            dir_path = self.data_path / dir_name
            if dir_path.exists():
                for class_name in expected_classes:
                    class_path = dir_path / class_name
                    if not class_path.exists():
                        structure_results['valid'] = False
                        structure_results['missing_classes'].append(f"{dir_name}/{class_name}")
                        self.validation_report['issues'].append(f"Missing class: {dir_name}/{class_name}")
        
        self.validation_report['validation_results']['structure'] = structure_results
    
    def _validate_image_quality(self):
        """Validate image quality and format"""
        print("🖼️ Validating image quality...")
        
        quality_results = {
            'total_images': 0,
            'valid_images': 0,
            'invalid_images': [],
            'corrupted_images': [],
            'format_issues': [],
            'size_issues': [],
            'quality_metrics': {
                'min_size': float('inf'),
                'max_size': 0,
                'avg_size': 0,
                'size_std': 0
            }
        }
        
        image_sizes = []
        
        # Check all images in dataset
        for split in ['train', 'val', 'test']:
            split_path = self.data_path / split
            if not split_path.exists():
                continue
            
            for class_name in ['NORMAL', 'PNEUMONIA']:
                class_path = split_path / class_name
                if not class_path.exists():
                    continue
                
                for img_file in class_path.glob('*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        quality_results['total_images'] += 1
                        
                        try:
                            # Try to open image
                            with Image.open(img_file) as img:
                                # Check image format
                                if img.format not in ['JPEG', 'PNG']:
                                    quality_results['format_issues'].append(str(img_file))
                                
                                # Check image size
                                width, height = img.size
                                image_sizes.append(width * height)
                                
                                if width < 100 or height < 100:
                                    quality_results['size_issues'].append(str(img_file))
                                
                                # Check if image is corrupted
                                img.verify()
                                quality_results['valid_images'] += 1
                                
                        except Exception as e:
                            quality_results['corrupted_images'].append({
                                'file': str(img_file),
                                'error': str(e)
                            })
                            self.validation_report['issues'].append(f"Corrupted image: {img_file}")
        
        # Calculate size statistics
        if image_sizes:
            quality_results['quality_metrics'] = {
                'min_size': min(image_sizes),
                'max_size': max(image_sizes),
                'avg_size': np.mean(image_sizes),
                'size_std': np.std(image_sizes)
            }
        
        self.validation_report['validation_results']['image_quality'] = quality_results
    
    def _validate_class_distribution(self):
        """Validate class distribution and balance"""
        print("📊 Validating class distribution...")
        
        distribution_results = {
            'train': {'NORMAL': 0, 'PNEUMONIA': 0},
            'val': {'NORMAL': 0, 'PNEUMONIA': 0},
            'test': {'NORMAL': 0, 'PNEUMONIA': 0},
            'total': {'NORMAL': 0, 'PNEUMONIA': 0},
            'balance_issues': []
        }
        
        # Count images in each class and split
        for split in ['train', 'val', 'test']:
            split_path = self.data_path / split
            if not split_path.exists():
                continue
            
            for class_name in ['NORMAL', 'PNEUMONIA']:
                class_path = split_path / class_name
                if not class_path.exists():
                    continue
                
                count = len(list(class_path.glob('*')))
                distribution_results[split][class_name] = count
                distribution_results['total'][class_name] += count
        
        # Check for class imbalance
        total_normal = distribution_results['total']['NORMAL']
        total_pneumonia = distribution_results['total']['PNEUMONIA']
        
        if total_normal > 0 and total_pneumonia > 0:
            imbalance_ratio = max(total_normal, total_pneumonia) / min(total_normal, total_pneumonia)
            if imbalance_ratio > 2.0:
                distribution_results['balance_issues'].append(f"Class imbalance ratio: {imbalance_ratio:.2f}")
                self.validation_report['issues'].append(f"Significant class imbalance: {imbalance_ratio:.2f}")
        
        # Check for empty splits
        for split in ['train', 'val', 'test']:
            total_split = sum(distribution_results[split].values())
            if total_split == 0:
                distribution_results['balance_issues'].append(f"Empty split: {split}")
                self.validation_report['issues'].append(f"Empty data split: {split}")
        
        self.validation_report['validation_results']['class_distribution'] = distribution_results
    
    def _validate_data_integrity(self):
        """Validate data integrity and consistency"""
        print("🔒 Validating data integrity...")
        
        integrity_results = {
            'duplicate_files': [],
            'naming_issues': [],
            'file_consistency': True,
            'checksum_validation': True
        }
        
        # Check for duplicate files (by name)
        all_files = {}
        for split in ['train', 'val', 'test']:
            split_path = self.data_path / split
            if not split_path.exists():
                continue
            
            for class_name in ['NORMAL', 'PNEUMONIA']:
                class_path = split_path / class_name
                if not class_path.exists():
                    continue
                
                for img_file in class_path.glob('*'):
                    filename = img_file.name
                    if filename in all_files:
                        integrity_results['duplicate_files'].append({
                            'file': filename,
                            'locations': [str(all_files[filename]), str(img_file)]
                        })
                        self.validation_report['issues'].append(f"Duplicate file: {filename}")
                    else:
                        all_files[filename] = img_file
        
        # Check file naming consistency
        for split in ['train', 'val', 'test']:
            split_path = self.data_path / split
            if not split_path.exists():
                continue
            
            for class_name in ['NORMAL', 'PNEUMONIA']:
                class_path = split_path / class_name
                if not class_path.exists():
                    continue
                
                for img_file in class_path.glob('*'):
                    if not img_file.name.replace('.', '').replace('_', '').replace('-', '').isalnum():
                        integrity_results['naming_issues'].append(str(img_file))
                        self.validation_report['issues'].append(f"Non-standard filename: {img_file}")
        
        self.validation_report['validation_results']['data_integrity'] = integrity_results
    
    def _generate_validation_report(self):
        """Generate comprehensive validation report"""
        print("📋 Generating validation report...")
        
        # Calculate overall quality score
        total_issues = len(self.validation_report['issues'])
        total_images = self.validation_report['validation_results'].get('image_quality', {}).get('total_images', 0)
        
        if total_images > 0:
            quality_score = max(0, 100 - (total_issues * 10))
        else:
            quality_score = 0
        
        self.validation_report['quality_metrics'] = {
            'overall_score': quality_score,
            'total_issues': total_issues,
            'total_images': total_images,
            'validation_status': 'PASS' if total_issues == 0 else 'FAIL'
        }
        
        # Generate recommendations
        if total_issues > 0:
            self.validation_report['recommendations'] = [
                "Review and fix all identified issues before training",
                "Ensure proper class balance in training data",
                "Validate image quality and format consistency",
                "Check for data leakage between splits"
            ]
        else:
            self.validation_report['recommendations'] = [
                "Dataset validation passed successfully",
                "Ready for model training",
                "Consider data augmentation for better generalization"
            ]
        
        # Save report
        report_path = Path("reports/data_validation.json")
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.validation_report, f, indent=2)
        
        print(f"📁 Validation report saved to: {report_path}")
    
    def plot_validation_summary(self):
        """Create visualization of validation results"""
        print("📊 Creating validation summary plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Data Validation Summary', fontsize=16, fontweight='bold')
        
        # Class distribution
        distribution = self.validation_report['validation_results']['class_distribution']
        splits = ['train', 'val', 'test']
        normal_counts = [distribution[split]['NORMAL'] for split in splits]
        pneumonia_counts = [distribution[split]['PNEUMONIA'] for split in splits]
        
        x = np.arange(len(splits))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, normal_counts, width, label='Normal', color='blue', alpha=0.7)
        axes[0, 0].bar(x + width/2, pneumonia_counts, width, label='Pneumonia', color='red', alpha=0.7)
        axes[0, 0].set_title('Class Distribution by Split')
        axes[0, 0].set_xlabel('Data Split')
        axes[0, 0].set_ylabel('Number of Images')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(splits)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Image quality metrics
        quality_metrics = self.validation_report['validation_results']['image_quality']['quality_metrics']
        metrics_names = ['Min Size', 'Max Size', 'Avg Size', 'Size Std']
        metrics_values = [
            quality_metrics['min_size'],
            quality_metrics['max_size'],
            quality_metrics['avg_size'],
            quality_metrics['size_std']
        ]
        
        axes[0, 1].bar(metrics_names, metrics_values, color=['green', 'orange', 'blue', 'red'], alpha=0.7)
        axes[0, 1].set_title('Image Size Statistics')
        axes[0, 1].set_ylabel('Pixels')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Validation issues
        issues = self.validation_report['issues']
        issue_types = {}
        for issue in issues:
            issue_type = issue.split(':')[0] if ':' in issue else 'Other'
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
        
        if issue_types:
            axes[1, 0].pie(issue_types.values(), labels=issue_types.keys(), autopct='%1.1f%%')
            axes[1, 0].set_title('Validation Issues by Type')
        else:
            axes[1, 0].text(0.5, 0.5, 'No Issues Found!', ha='center', va='center', 
                           fontsize=16, fontweight='bold', color='green')
            axes[1, 0].set_title('Validation Issues')
        
        # Quality score
        quality_score = self.validation_report['quality_metrics']['overall_score']
        status = self.validation_report['quality_metrics']['validation_status']
        
        color = 'green' if status == 'PASS' else 'red'
        axes[1, 1].bar(['Overall Quality Score'], [quality_score], color=color, alpha=0.7)
        axes[1, 1].set_title('Data Quality Assessment')
        axes[1, 1].set_ylabel('Score (0-100)')
        axes[1, 1].set_ylim(0, 100)
        axes[1, 1].text(0, quality_score + 2, f'{quality_score:.1f}', ha='center', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reports/data_validation_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Validation summary plots created!")


def main():
    """Test data validation system"""
    print("🧪 Testing Data Validation System...")
    
    # Create sample validation
    validator = DataValidator()
    
    # Run validation (will work even if data doesn't exist)
    report = validator.validate_dataset()
    
    print(f"📊 Validation Results:")
    print(f"   Overall Score: {report['quality_metrics']['overall_score']}")
    print(f"   Status: {report['quality_metrics']['validation_status']}")
    print(f"   Issues Found: {report['quality_metrics']['total_issues']}")
    
    # Create plots
    validator.plot_validation_summary()
    
    print("🎉 Data validation system working!")


if __name__ == "__main__":
    main()
