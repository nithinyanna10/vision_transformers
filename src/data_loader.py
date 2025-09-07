"""
Advanced Data Loading Pipeline for Medical ViT
Includes custom augmentations, class balancing, and efficient loading
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import cv2


class MedicalImageDataset(Dataset):
    """Custom dataset for medical images with advanced augmentations"""
    
    def __init__(self, image_paths, labels, transform=None, is_training=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_training = is_training
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return image, label


class DataManager:
    """Manages data loading, splitting, and augmentation"""
    
    def __init__(self, config):
        self.config = config
        self.dataset_path = config['data']['dataset_path']
        
    def get_transforms(self):
        """Get training and validation transforms"""
        
        # Training transforms with medical-specific augmentations
        train_transform = A.Compose([
            A.Resize(self.config['data']['img_size'], self.config['data']['img_size']),
            A.HorizontalFlip(p=self.config['augmentation']['horizontal_flip']),
            A.Rotate(limit=self.config['augmentation']['rotation'], p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=self.config['augmentation']['brightness_contrast'],
                contrast_limit=self.config['augmentation']['brightness_contrast'],
                p=0.5
            ),
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                p=0.3 if self.config['augmentation']['elastic_transform'] else 0
            ),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.Normalize(
                mean=self.config['data']['mean'],
                std=self.config['data']['std']
            ),
            ToTensorV2()
        ])
        
        # Validation transforms (no augmentation)
        val_transform = A.Compose([
            A.Resize(self.config['data']['img_size'], self.config['data']['img_size']),
            A.Normalize(
                mean=self.config['data']['mean'],
                std=self.config['data']['std']
            ),
            ToTensorV2()
        ])
        
        return train_transform, val_transform
    
    def load_data(self):
        """Load and organize data from directory structure"""
        print("Loading data from:", self.dataset_path)
        
        image_paths = []
        labels = []
        class_names = []
        
        # Expected structure: data/chest_xray/train/NORMAL, data/chest_xray/train/PNEUMONIA
        for split in ['train', 'test']:
            split_path = os.path.join(self.dataset_path, split)
            if not os.path.exists(split_path):
                print(f"Warning: {split_path} not found")
                continue
                
            for class_name in os.listdir(split_path):
                class_path = os.path.join(split_path, class_name)
                if not os.path.isdir(class_path):
                    continue
                    
                class_names.append(class_name)
                class_label = 0 if class_name == 'NORMAL' else 1
                
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, img_file)
                        image_paths.append(img_path)
                        labels.append(class_label)
        
        print(f"Total images loaded: {len(image_paths)}")
        print(f"Class distribution: {Counter(labels)}")
        
        return image_paths, labels, list(set(class_names))
    
    def create_data_splits(self, image_paths, labels):
        """Create train/val/test splits"""
        
        # First split: train+val vs test
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            image_paths, labels,
            test_size=self.config['data']['test_split'],
            stratify=labels,
            random_state=42
        )
        
        # Second split: train vs val
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels,
            test_size=self.config['data']['val_split'] / (1 - self.config['data']['test_split']),
            stratify=train_val_labels,
            random_state=42
        )
        
        print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
        
        return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)
    
    def get_class_weights(self, labels):
        """Calculate class weights for imbalanced dataset"""
        class_counts = Counter(labels)
        total_samples = len(labels)
        
        weights = []
        for label in sorted(class_counts.keys()):
            weight = total_samples / (len(class_counts) * class_counts[label])
            weights.append(weight)
        
        return torch.FloatTensor(weights)
    
    def create_dataloaders(self):
        """Create train, validation, and test dataloaders"""
        
        # Load data
        image_paths, labels, class_names = self.load_data()
        
        # Create splits
        (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = self.create_data_splits(image_paths, labels)
        
        # Get transforms
        train_transform, val_transform = self.get_transforms()
        
        # Create datasets
        train_dataset = MedicalImageDataset(train_paths, train_labels, train_transform, is_training=True)
        val_dataset = MedicalImageDataset(val_paths, val_labels, val_transform, is_training=False)
        test_dataset = MedicalImageDataset(test_paths, test_labels, val_transform, is_training=False)
        
        # Calculate class weights for weighted sampling
        class_weights = self.get_class_weights(train_labels)
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['data']['batch_size'],
            sampler=sampler,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader, class_names


def mixup_data(x, y, alpha=1.0):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
