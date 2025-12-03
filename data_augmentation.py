"""
Advanced Data Augmentation Module for EV Car Classification
Uses Albumentations library for high-quality augmentations
"""

import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import random
from typing import Tuple, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAugmentor:
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize data augmentor with advanced transformations
        
        Args:
            target_size: Target image size (height, width)
        """
        self.target_size = target_size
        
        # Training augmentations - aggressive but realistic
        self.train_transform = A.Compose([
            # Resize and maintain aspect ratio
            A.LongestMaxSize(max_size=max(target_size), p=1.0),
            A.PadIfNeeded(min_height=target_size[0], min_width=target_size[1], 
                         border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
            
            # Geometric transformations
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=10, p=0.5),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            
            # Color and lighting augmentations
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            
            # Weather and environmental effects
            A.OneOf([
                A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=1, p=0.3),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, p=0.2),
                A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, p=0.2),
            ], p=0.3),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
                A.MultiplicativeNoise(multiplier=[0.9, 1.1], elementwise=True, p=0.2),
            ], p=0.3),
            
            A.OneOf([
                A.GaussianBlur(blur_limit=(1, 3), p=0.3),
                A.MotionBlur(blur_limit=3, p=0.3),
            ], p=0.2),
            
            # Occlusion simulation
            A.CoarseDropout(max_holes=3, max_height=32, max_width=32, 
                           fill_value=0, mask_fill_value=0, p=0.2),
            
            # Normalize
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Validation/Test transformations - minimal preprocessing
        self.val_transform = A.Compose([
            A.LongestMaxSize(max_size=max(target_size), p=1.0),
            A.PadIfNeeded(min_height=target_size[0], min_width=target_size[1], 
                         border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Inference transformations
        self.inference_transform = A.Compose([
            A.LongestMaxSize(max_size=max(target_size), p=1.0),
            A.PadIfNeeded(min_height=target_size[0], min_width=target_size[1], 
                         border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def augment_dataset(self, input_dir: str, output_dir: str, 
                       augmentation_factor: int = 5) -> Dict[str, int]:
        """
        Augment dataset to increase size and diversity
        
        Args:
            input_dir: Directory with EV and Non-EV folders
            output_dir: Directory to save augmented dataset
            augmentation_factor: Number of augmented versions per original image
            
        Returns:
            Dictionary with augmentation statistics
        """
        stats = {'original_images': 0, 'augmented_images': 0}
        
        os.makedirs(output_dir, exist_ok=True)
        
        for category in ['EV', 'Non-EV']:
            input_category_dir = os.path.join(input_dir, category)
            output_category_dir = os.path.join(output_dir, category)
            
            if not os.path.exists(input_category_dir):
                logger.warning(f"Input directory not found: {input_category_dir}")
                continue
                
            os.makedirs(output_category_dir, exist_ok=True)
            
            # Get all image files
            image_files = [f for f in os.listdir(input_category_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            logger.info(f"Augmenting {len(image_files)} images in {category} category")
            
            for img_file in image_files:
                img_path = os.path.join(input_category_dir, img_file)
                base_name = os.path.splitext(img_file)[0]
                
                try:
                    # Load original image
                    image = cv2.imread(img_path)
                    if image is None:
                        logger.warning(f"Could not load image: {img_path}")
                        continue
                    
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    stats['original_images'] += 1
                    
                    # Save original image
                    original_output_path = os.path.join(output_category_dir, f"{base_name}_original.jpg")
                    cv2.imwrite(original_output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    
                    # Create augmented versions
                    for aug_idx in range(augmentation_factor):
                        try:
                            # Apply augmentation (excluding normalization and tensor conversion)
                            aug_transform = A.Compose([
                                A.LongestMaxSize(max_size=max(self.target_size), p=1.0),
                                A.PadIfNeeded(min_height=self.target_size[0], min_width=self.target_size[1], 
                                             border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
                                A.HorizontalFlip(p=0.5),
                                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=10, p=0.5),
                                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
                                A.OneOf([
                                    A.GaussNoise(var_limit=(10, 30), p=0.3),
                                    A.GaussianBlur(blur_limit=(1, 3), p=0.3),
                                ], p=0.3),
                            ])
                            
                            augmented = aug_transform(image=image)['image']
                            
                            # Save augmented image
                            aug_filename = f"{base_name}_aug_{aug_idx}.jpg"
                            aug_output_path = os.path.join(output_category_dir, aug_filename)
                            cv2.imwrite(aug_output_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
                            stats['augmented_images'] += 1
                            
                        except Exception as e:
                            logger.error(f"Failed to augment {img_file} (iteration {aug_idx}): {str(e)}")
                
                except Exception as e:
                    logger.error(f"Failed to process {img_path}: {str(e)}")
        
        logger.info(f"Augmentation completed: {stats}")
        return stats
    
    def get_transforms(self, mode: str = 'train'):
        """
        Get appropriate transforms for different modes
        
        Args:
            mode: 'train', 'val', or 'inference'
            
        Returns:
            Albumentations transform pipeline
        """
        if mode == 'train':
            return self.train_transform
        elif mode == 'val':
            return self.val_transform
        elif mode == 'inference':
            return self.inference_transform
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'train', 'val', or 'inference'")
    
    def preview_augmentations(self, image_path: str, num_samples: int = 5, 
                            output_dir: str = None):
        """
        Preview augmentation effects on a single image
        
        Args:
            image_path: Path to input image
            num_samples: Number of augmented samples to generate
            output_dir: Directory to save preview images
        """
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            # Save original
            original_path = os.path.join(output_dir, f"{base_name}_original.jpg")
            cv2.imwrite(original_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Create preview transform (without normalization)
        preview_transform = A.Compose([
            A.LongestMaxSize(max_size=max(self.target_size), p=1.0),
            A.PadIfNeeded(min_height=self.target_size[0], min_width=self.target_size[1], 
                         border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=10, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.7),
            A.OneOf([
                A.GaussNoise(var_limit=(10, 30), p=0.5),
                A.GaussianBlur(blur_limit=(1, 3), p=0.5),
            ], p=0.5),
        ])
        
        for i in range(num_samples):
            augmented_image = preview_transform(image=image)['image']
            
            if output_dir:
                preview_path = os.path.join(output_dir, f"{base_name}_preview_{i}.jpg")
                cv2.imwrite(preview_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
            else:
                # Display using matplotlib if available
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(8, 6))
                    plt.imshow(augmented_image)
                    plt.title(f"Augmented Sample {i+1}")
                    plt.axis('off')
                    plt.show()
                except ImportError:
                    logger.warning("Matplotlib not available for preview display")

if __name__ == "__main__":
    # Example usage
    augmentor = DataAugmentor(target_size=(224, 224))
    
    # Augment dataset
    input_dir = "processed_dataset"  # Output from car_detector.py
    output_dir = "augmented_dataset"
    augmentation_factor = 3  # Create 3 augmented versions per original image
    
    if os.path.exists(input_dir):
        stats = augmentor.augment_dataset(input_dir, output_dir, augmentation_factor)
        print(f"Dataset augmentation completed: {stats}")
        
        # Preview augmentations on a sample image
        sample_images = []
        for category in ['EV', 'Non-EV']:
            category_dir = os.path.join(input_dir, category)
            if os.path.exists(category_dir):
                images = [f for f in os.listdir(category_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    sample_images.append(os.path.join(category_dir, images[0]))
        
        if sample_images:
            print("Generating augmentation previews...")
            augmentor.preview_augmentations(sample_images[0], num_samples=5, 
                                          output_dir="augmentation_preview")
    else:
        print("Processed dataset directory not found. Run car_detector.py first.")