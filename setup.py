"""
Setup Script for Indian EV Car Detection System
Handles dataset preparation, model training pipeline, and system configuration
"""

import os
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directory_structure():
    """Create necessary directory structure"""
    directories = [
        'dataset/EV',
        'dataset/Non-EV', 
        'processed_dataset/EV',
        'processed_dataset/Non-EV',
        'augmented_dataset/EV',
        'augmented_dataset/Non-EV',
        'models',
        'results',
        'logs',
        'test_images',
        'output'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'torch', 'torchvision', 'ultralytics', 'cv2',
        'PIL', 'numpy', 'matplotlib', 'sklearn',
        'albumentations', 'timm', 'tensorboard', 'tqdm', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"✗ {package} is missing")
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Please install missing packages using: pip install -r requirements.txt")
        return False
    
    logger.info("All dependencies are installed!")
    return True

def setup_dataset():
    """Setup and prepare dataset"""
    logger.info("Setting up dataset...")
    
    # Check if original dataset exists
    ev_dir = Path('dataset/EV')
    non_ev_dir = Path('dataset/Non-EV')
    
    if not ev_dir.exists() or not non_ev_dir.exists():
        logger.warning("Original dataset directories not found!")
        logger.info("Please place your images in:")
        logger.info("  - dataset/EV/     (for electric vehicle images)")
        logger.info("  - dataset/Non-EV/ (for non-electric vehicle images)")
        return False
    
    # Count images
    ev_images = list(ev_dir.glob('*.jpg')) + list(ev_dir.glob('*.jpeg')) + list(ev_dir.glob('*.png'))
    non_ev_images = list(non_ev_dir.glob('*.jpg')) + list(non_ev_dir.glob('*.jpeg')) + list(non_ev_dir.glob('*.png'))
    
    logger.info(f"Found {len(ev_images)} EV images")
    logger.info(f"Found {len(non_ev_images)} Non-EV images")
    logger.info(f"Total images: {len(ev_images) + len(non_ev_images)}")
    
    if len(ev_images) == 0 or len(non_ev_images) == 0:
        logger.error("Dataset is incomplete. Both EV and Non-EV categories need images.")
        return False
    
    return True

def run_car_detection():
    """Run car detection and cropping"""
    logger.info("Starting car detection and cropping...")
    
    try:
        from car_detector import CarDetector
        
        detector = CarDetector(model_name='yolov8s.pt', confidence_threshold=0.5)
        
        # Process dataset
        stats = detector.process_dataset('dataset', 'processed_dataset')
        
        logger.info(f"Car detection completed: {stats}")
        return True
        
    except Exception as e:
        logger.error(f"Car detection failed: {str(e)}")
        return False

def run_data_augmentation():
    """Run data augmentation"""
    logger.info("Starting data augmentation...")
    
    try:
        from data_augmentation import DataAugmentor
        
        augmentor = DataAugmentor(target_size=(224, 224))
        
        # Augment dataset
        stats = augmentor.augment_dataset('processed_dataset', 'augmented_dataset', augmentation_factor=3)
        
        logger.info(f"Data augmentation completed: {stats}")
        return True
        
    except Exception as e:
        logger.error(f"Data augmentation failed: {str(e)}")
        return False

def run_training():
    """Run model training"""
    logger.info("Starting model training...")
    
    try:
        from main import main as train_main
        
        # Run training
        train_main()
        
        logger.info("Model training completed!")
        return True
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        return False

def create_sample_config():
    """Create sample configuration file"""
    config = {
        "model": {
            "architecture": "efficientnet_b3",
            "num_classes": 2,
            "dropout_rate": 0.3,
            "use_attention": True
        },
        "training": {
            "batch_size": 32,
            "epochs": 50,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "loss_function": "focal"
        },
        "data": {
            "image_size": [224, 224],
            "augmentation_factor": 3,
            "train_val_split": 0.8
        },
        "detection": {
            "yolo_model": "yolov8s.pt",
            "confidence_threshold": 0.5
        }
    }
    
    import json
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("Sample configuration saved to config.json")

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description='Setup Indian EV Car Detection System')
    parser.add_argument('--step', choices=['all', 'deps', 'dirs', 'dataset', 'detect', 'augment', 'train'], 
                       default='all', help='Setup step to run')
    parser.add_argument('--skip-deps', action='store_true', help='Skip dependency check')
    
    args = parser.parse_args()
    
    logger.info("🚗 Indian EV Car Detection System Setup 🚗")
    logger.info("=" * 50)
    
    # Step 1: Check dependencies
    if args.step in ['all', 'deps'] and not args.skip_deps:
        logger.info("Step 1: Checking dependencies...")
        if not check_dependencies():
            logger.error("Dependency check failed. Please install required packages.")
            return
    
    # Step 2: Create directory structure
    if args.step in ['all', 'dirs']:
        logger.info("Step 2: Creating directory structure...")
        create_directory_structure()
    
    # Step 3: Setup dataset
    if args.step in ['all', 'dataset']:
        logger.info("Step 3: Setting up dataset...")
        if not setup_dataset():
            logger.error("Dataset setup failed. Please check your images.")
            return
    
    # Step 4: Run car detection
    if args.step in ['all', 'detect']:
        logger.info("Step 4: Running car detection...")
        if not run_car_detection():
            logger.error("Car detection failed.")
            return
    
    # Step 5: Run data augmentation
    if args.step in ['all', 'augment']:
        logger.info("Step 5: Running data augmentation...")
        if not run_data_augmentation():
            logger.error("Data augmentation failed.")
            return
    
    # Step 6: Run training
    if args.step in ['all', 'train']:
        logger.info("Step 6: Running model training...")
        if not run_training():
            logger.error("Model training failed.")
            return
    
    # Create sample configuration
    create_sample_config()
    
    logger.info("=" * 50)
    logger.info("✅ Setup completed successfully!")
    logger.info("=" * 50)
    
    logger.info("Next steps:")
    logger.info("1. Run the GUI application: python ui_app.py")
    logger.info("2. Or use inference directly: python inference.py")
    logger.info("3. Check results in the 'results' directory")
    logger.info("4. Modify config.json for custom settings")

if __name__ == "__main__":
    main()