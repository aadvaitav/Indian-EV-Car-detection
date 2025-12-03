"""
Filename Fixer for Dataset
Fixes problematic filenames that cause OpenCV errors
"""

import os
import re
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_filename(filename):
    """
    Clean filename by removing/replacing problematic characters
    
    Args:
        filename: Original filename
        
    Returns:
        Cleaned filename safe for OpenCV
    """
    # Get file extension
    name, ext = os.path.splitext(filename)
    
    # Replace problematic characters
    cleaned_name = name
    
    # Replace common problematic characters
    replacements = {
        '»': '_',
        '«': '_',
        '"': '_',
        '"': '_',
        ''': '_',
        ''': '_',
        '–': '-',
        '—': '-',
        '…': '_',
        'ΓÇ»': '_',
        'ΓÇö': '-',
        'ΓÇÖ': '_',
        'ΓÇØ': '_',
        'ΓÇ¥': '_',
        'ΓÇ£': '_',
        'ΓÇ¥': '_'
    }
    
    for old_char, new_char in replacements.items():
        cleaned_name = cleaned_name.replace(old_char, new_char)
    
    # Remove other non-ASCII characters
    cleaned_name = re.sub(r'[^\x00-\x7F]+', '_', cleaned_name)
    
    # Replace multiple consecutive underscores/hyphens with single ones
    cleaned_name = re.sub(r'[-_]+', '_', cleaned_name)
    
    # Remove leading/trailing underscores
    cleaned_name = cleaned_name.strip('_-')
    
    # Ensure filename is not empty
    if not cleaned_name:
        cleaned_name = "image"
    
    return cleaned_name + ext

def fix_dataset_filenames(dataset_dir):
    """
    Fix all problematic filenames in dataset directory
    
    Args:
        dataset_dir: Path to dataset directory
    """
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return
    
    fixed_count = 0
    error_count = 0
    
    # Process all subdirectories
    for category_dir in dataset_path.iterdir():
        if not category_dir.is_dir():
            continue
            
        logger.info(f"Processing category: {category_dir.name}")
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(category_dir.glob(f'*{ext}'))
            image_files.extend(category_dir.glob(f'*{ext.upper()}'))
        
        logger.info(f"Found {len(image_files)} images in {category_dir.name}")
        
        for image_file in image_files:
            original_name = image_file.name
            cleaned_name = clean_filename(original_name)
            
            if original_name != cleaned_name:
                try:
                    new_path = image_file.parent / cleaned_name
                    
                    # Handle duplicate names
                    counter = 1
                    base_name, ext = os.path.splitext(cleaned_name)
                    while new_path.exists():
                        cleaned_name = f"{base_name}_{counter}{ext}"
                        new_path = image_file.parent / cleaned_name
                        counter += 1
                    
                    # Rename the file
                    image_file.rename(new_path)
                    logger.info(f"Renamed: {original_name} -> {cleaned_name}")
                    fixed_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to rename {original_name}: {str(e)}")
                    error_count += 1
    
    logger.info(f"Filename fixing completed!")
    logger.info(f"Files renamed: {fixed_count}")
    logger.info(f"Errors: {error_count}")

def check_problematic_files(dataset_dir):
    """
    Check for files that might cause issues
    
    Args:
        dataset_dir: Path to dataset directory
    """
    dataset_path = Path(dataset_dir)
    problematic_files = []
    
    if not dataset_path.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return []
    
    # Check all files
    for category_dir in dataset_path.iterdir():
        if not category_dir.is_dir():
            continue
            
        for file_path in category_dir.iterdir():
            if not file_path.is_file():
                continue
                
            filename = file_path.name
            
            # Check for problematic characters
            problematic_chars = ['»', '«', '"', '"', ''', ''', '–', '—', '…', 'ΓÇ']
            
            has_issues = False
            issues = []
            
            # Check for non-ASCII characters
            if not filename.isascii():
                has_issues = True
                issues.append("non-ASCII characters")
            
            # Check for specific problematic characters
            for char in problematic_chars:
                if char in filename:
                    has_issues = True
                    issues.append(f"contains '{char}'")
            
            # Check if file can be read by OpenCV
            import cv2
            try:
                img = cv2.imread(str(file_path))
                if img is None:
                    has_issues = True
                    issues.append("cannot be read by OpenCV")
            except Exception as e:
                has_issues = True
                issues.append(f"OpenCV error: {str(e)}")
            
            if has_issues:
                problematic_files.append({
                    'path': str(file_path),
                    'filename': filename,
                    'issues': issues
                })
    
    return problematic_files

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix problematic filenames in dataset')
    parser.add_argument('--dataset', default='augmented_dataset', 
                       help='Dataset directory path')
    parser.add_argument('--check-only', action='store_true', 
                       help='Only check for problematic files, do not fix')
    
    args = parser.parse_args()
    
    if args.check_only:
        logger.info("Checking for problematic files...")
        problematic_files = check_problematic_files(args.dataset)
        
        if problematic_files:
            logger.warning(f"Found {len(problematic_files)} problematic files:")
            for file_info in problematic_files[:10]:  # Show first 10
                logger.warning(f"  {file_info['filename']}: {', '.join(file_info['issues'])}")
            if len(problematic_files) > 10:
                logger.warning(f"  ... and {len(problematic_files) - 10} more")
        else:
            logger.info("No problematic files found!")
    else:
        logger.info("Fixing problematic filenames...")
        fix_dataset_filenames(args.dataset)

if __name__ == "__main__":
    main()