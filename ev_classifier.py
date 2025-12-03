"""
Advanced EV Car Classification Model
Uses state-of-the-art CNN architecture with attention mechanisms and regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, Tuple
import math

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel attention"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        # Channel attention
        self.channel_attention = SEBlock(channels, reduction)
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel attention
        x = self.channel_attention(x)
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_attention(spatial_att)
        
        return x * spatial_att

class EVClassifier(nn.Module):
    """
    Advanced EV Car Classifier using EfficientNet backbone with attention mechanisms
    """
    def __init__(self, 
                 model_name: str = 'efficientnet_b3',
                 num_classes: int = 2,
                 dropout_rate: float = 0.3,
                 use_attention: bool = True):
        super().__init__()
        
        # Load pre-trained backbone
        self.backbone = timm.create_model(
            model_name, 
            pretrained=True, 
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling
        )
        
        # Get feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]
            self.feature_size = features.shape[2:]
        
        # Attention mechanism
        self.use_attention = use_attention
        if use_attention:
            self.attention = CBAM(self.feature_dim)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head with advanced regularization
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.25),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features using backbone
        features = self.backbone(x)
        
        # Apply attention if enabled
        if self.use_attention:
            features = self.attention(features)
        
        # Global average pooling
        features = self.global_pool(features)
        features = features.view(features.size(0), -1)
        
        # Classification
        logits = self.classifier(features)
        
        return logits
    
    def extract_features(self, x):
        """Extract features for visualization or analysis"""
        with torch.no_grad():
            features = self.backbone(x)
            if self.use_attention:
                features = self.attention(features)
            pooled_features = self.global_pool(features)
            return pooled_features.view(pooled_features.size(0), -1)

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing for better generalization
    """
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        pred = F.log_softmax(pred, dim=1)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=1))

def create_model(model_config: dict = None) -> EVClassifier:
    """
    Factory function to create EV classifier model
    
    Args:
        model_config: Dictionary with model configuration
        
    Returns:
        EVClassifier instance
    """
    if model_config is None:
        model_config = {
            'model_name': 'efficientnet_b3',
            'num_classes': 2,
            'dropout_rate': 0.3,
            'use_attention': True
        }
    
    return EVClassifier(**model_config)

def get_model_variants():
    """
    Get different model variants for experimentation
    
    Returns:
        Dictionary of model configurations
    """
    return {
        'efficient_light': {
            'model_name': 'efficientnet_b0',
            'num_classes': 2,
            'dropout_rate': 0.2,
            'use_attention': False
        },
        'efficient_standard': {
            'model_name': 'efficientnet_b3',
            'num_classes': 2,
            'dropout_rate': 0.3,
            'use_attention': True
        },
        'efficient_heavy': {
            'model_name': 'efficientnet_b5',
            'num_classes': 2,
            'dropout_rate': 0.4,
            'use_attention': True
        },
        'resnet_standard': {
            'model_name': 'resnet50',
            'num_classes': 2,
            'dropout_rate': 0.3,
            'use_attention': True
        },
        'convnext_standard': {
            'model_name': 'convnext_base',
            'num_classes': 2,
            'dropout_rate': 0.3,
            'use_attention': True
        }
    }

if __name__ == "__main__":
    # Test model creation and forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_model()
    model = model.to(device)
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    
    with torch.no_grad():
        outputs = model(dummy_input)
        print(f"Model output shape: {outputs.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test feature extraction
        features = model.extract_features(dummy_input)
        print(f"Feature shape: {features.shape}")
    
    # Test different model variants
    print("\nAvailable model variants:")
    variants = get_model_variants()
    for name, config in variants.items():
        print(f"  {name}: {config['model_name']} (attention: {config['use_attention']})")
    
    print("\nModel created successfully!")