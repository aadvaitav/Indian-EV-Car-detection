"""
Main Training Script for EV Car Classification
Handles dataset loading, training, validation, and model saving with advanced techniques
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import json
import logging
from tqdm import tqdm
import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from ev_classifier import create_model, get_model_variants, FocalLoss, LabelSmoothingLoss
from data_augmentation import DataAugmentor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EVDataset(Dataset):
    """Custom dataset for EV car classification"""
    
    def __init__(self, data_dir, transform=None, split='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        
        # Class mapping
        self.class_to_idx = {'Non-EV': 0, 'EV': 1}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Load data
        self.samples = self._load_samples()
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
        
        # Print class distribution
        class_counts = defaultdict(int)
        for _, label in self.samples:
            class_counts[self.idx_to_class[label]] += 1
        logger.info(f"Class distribution: {dict(class_counts)}")
    
    def _load_samples(self):
        """Load all samples with their labels"""
        samples = []
        
        for class_name in ['EV', 'Non-EV']:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                logger.warning(f"Directory not found: {class_dir}")
                continue
            
            # Get all image files
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                label = self.class_to_idx[class_name]
                samples.append((img_path, label))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image with robust error handling
        image = None
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Try to load image
                image = cv2.imread(img_path)
                
                if image is None:
                    # If image is None, try different approach
                    import numpy as np
                    from PIL import Image as PILImage
                    
                    # Try with PIL
                    pil_img = PILImage.open(img_path)
                    image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
                if image is not None:
                    # Convert BGR to RGB for processing
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    break
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {img_path}: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to load image after {max_retries} attempts: {img_path}")
                    # Create a dummy image as fallback
                    image = np.zeros((224, 224, 3), dtype=np.uint8)
                    break
        
        # Final fallback if image is still None
        if image is None:
            logger.error(f"Creating dummy image for: {img_path}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Apply transforms
        if self.transform:
            if isinstance(self.transform, A.Compose):
                try:
                    transformed = self.transform(image=image)
                    image = transformed['image']
                except Exception as e:
                    logger.error(f"Transform failed for {img_path}: {str(e)}")
                    # Apply minimal transform as fallback
                    import torch
                    from torchvision import transforms
                    fallback_transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    image = fallback_transform(image)
            else:
                image = Image.fromarray(image)
                image = self.transform(image)
        
        return image, label

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        
        return self.counter >= self.patience
    
    def save_checkpoint(self, model):
        """Save model checkpoint"""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()
    
    def restore(self, model):
        """Restore best weights"""
        if self.best_weights:
            model.load_state_dict(self.best_weights)

class Trainer:
    """Advanced training class with comprehensive features"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize data augmentor
        self.augmentor = DataAugmentor(target_size=(224, 224))
        
        # Setup datasets and dataloaders
        self._setup_data()
        
        # Setup model
        self._setup_model()
        
        # Setup training components
        self._setup_training()
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def _setup_data(self):
        """Setup datasets and data loaders"""
        data_dir = self.config['data_dir']
        
        # Create datasets
        train_dataset = EVDataset(
            data_dir, 
            transform=self.augmentor.get_transforms('train'),
            split='train'
        )
        
        # Split dataset
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        # Update val_dataset transform
        val_dataset.dataset.transform = self.augmentor.get_transforms('val')
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    def _setup_model(self):
        """Setup model architecture"""
        model_config = self.config['model']
        self.model = create_model(model_config)
        self.model = self.model.to(self.device)
        
        # Model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    def _setup_training(self):
        """Setup training components"""
        # Loss function
        if self.config['loss_function'] == 'focal':
            self.criterion = FocalLoss(alpha=1.0, gamma=2.0)
        elif self.config['loss_function'] == 'label_smoothing':
            self.criterion = LabelSmoothingLoss(num_classes=2, smoothing=0.1)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        if self.config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        
        # Learning rate scheduler
        if self.config['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config['epochs'] // 4,
                T_mult=2
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config['early_stopping_patience'],
            min_delta=0.001
        )
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_samples += target.size(0)
            correct_predictions += predicted.eq(target).sum().item()
            
            # Update progress bar
            accuracy = 100. * correct_predictions / total_samples
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{accuracy:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct_predictions / total_samples
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                loss = self.criterion(outputs, target)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total_samples += target.size(0)
                correct_predictions += predicted.eq(target).sum().item()
                
                # Store for metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                # Update progress bar
                accuracy = 100. * correct_predictions / total_samples
                pbar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.4f}',
                    'Acc': f'{accuracy:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct_predictions / total_samples
        
        return epoch_loss, epoch_acc, all_predictions, all_targets
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        start_time = time.time()
        
        best_val_acc = 0.0
        
        for epoch in range(self.config['epochs']):
            logger.info(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validate
            val_loss, val_acc, val_predictions, val_targets = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Update learning rate
            if self.config['scheduler'] == 'cosine':
                self.scheduler.step()
            else:
                self.scheduler.step(val_loss)
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            logger.info(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('best_model.pth')
                logger.info(f"New best validation accuracy: {best_val_acc:.2f}%")
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                logger.info("Early stopping triggered")
                break
        
        # Restore best weights
        self.early_stopping.restore(self.model)
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Final evaluation
        self.evaluate()
        
        # Save final model and training history
        self.save_model('final_model.pth')
        self.save_training_history()
        self.plot_training_history()
    
    def evaluate(self):
        """Comprehensive model evaluation"""
        logger.info("Evaluating model...")
        
        val_loss, val_acc, predictions, targets = self.validate()
        
        # Classification report
        report = classification_report(
            targets, predictions,
            target_names=['Non-EV', 'EV'],
            digits=4
        )
        logger.info(f"Classification Report:\n{report}")
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Non-EV', 'EV'],
                   yticklabels=['Non-EV', 'EV'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Additional metrics
        f1 = f1_score(targets, predictions, average='weighted')
        logger.info(f"F1 Score: {f1:.4f}")
    
    def save_model(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        torch.save(checkpoint, filename)
        logger.info(f"Model saved to {filename}")
    
    def save_training_history(self):
        """Save training history"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'config': self.config
        }
        
        with open('training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info("Training history saved to training_history.json")
    
    def plot_training_history(self):
        """Plot training history"""
        epochs = range(1, len(self.train_losses) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(epochs, self.train_losses, 'bo-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'ro-', label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epochs, self.train_accuracies, 'bo-', label='Training Accuracy')
        ax2.plot(epochs, self.val_accuracies, 'ro-', label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Training history plot saved to training_history.png")

def main():
    """Main training function"""
    # Training configuration
    config = {
        'data_dir': 'augmented_dataset',  # Directory with EV and Non-EV folders
        'model': {
            'model_name': 'efficientnet_b3',
            'num_classes': 2,
            'dropout_rate': 0.3,
            'use_attention': True
        },
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'optimizer': 'adamw',  # 'adam' or 'adamw'
        'scheduler': 'cosine',  # 'cosine' or 'plateau'
        'loss_function': 'focal',  # 'ce', 'focal', or 'label_smoothing'
        'early_stopping_patience': 10,
        'num_workers': 4
    }
    
    # Check if dataset exists
    if not os.path.exists(config['data_dir']):
        logger.error(f"Dataset directory not found: {config['data_dir']}")
        logger.info("Please run car_detector.py and data_augmentation.py first")
        return
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()