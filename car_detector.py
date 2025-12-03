"""
Car Detection Module using YOLOv8
Detects cars in images and extracts bounding boxes for EV classification
"""

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
from typing import List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CarDetector:
    def __init__(self, model_name: str = 'yolov8n.pt', confidence_threshold: float = 0.5):
        """
        Initialize car detector with YOLOv8 model
        
        Args:
            model_name: YOLOv8 model variant ('yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt')
            confidence_threshold: Minimum confidence for detections
        """
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        # COCO class IDs for vehicles
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck'
        }
        
    def detect_cars(self, image: np.ndarray) -> List[dict]:
        """
        Detect cars in image and return bounding boxes
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detection dictionaries containing bbox, confidence, and class
        """
        results = self.model(image, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Filter for vehicles with sufficient confidence
                    if class_id in self.vehicle_classes and confidence >= self.confidence_threshold:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class': self.vehicle_classes[class_id],
                            'class_id': class_id
                        }
                        detections.append(detection)
        
        return detections
    
    def extract_car_crops(self, image: np.ndarray, detections: List[dict]) -> List[np.ndarray]:
        """
        Extract cropped car images from detections
        
        Args:
            image: Original image
            detections: List of detection dictionaries
            
        Returns:
            List of cropped car images
        """
        crops = []
        h, w = image.shape[:2]
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # Extract crop with some padding
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
        
        return crops
    
    def process_image(self, image_path: str, save_crops: bool = False, 
                     output_dir: str = None) -> Tuple[List[np.ndarray], List[dict]]:
        """
        Process single image for car detection and cropping
        
        Args:
            image_path: Path to input image
            save_crops: Whether to save detected car crops
            output_dir: Directory to save crops
            
        Returns:
            Tuple of (car_crops, detections)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return [], []
        
        # Detect cars
        detections = self.detect_cars(image)
        logger.info(f"Detected {len(detections)} cars in {image_path}")
        
        # Extract crops
        crops = self.extract_car_crops(image, detections)
        
        # Save crops if requested
        if save_crops and output_dir and crops:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            for i, crop in enumerate(crops):
                crop_filename = f"{base_name}_car_{i}.jpg"
                crop_path = os.path.join(output_dir, crop_filename)
                cv2.imwrite(crop_path, crop)
        
        return crops, detections
    
    def process_dataset(self, dataset_dir: str, output_dir: str) -> dict:
        """
        Process entire dataset for car detection and cropping
        
        Args:
            dataset_dir: Directory containing EV and Non-EV folders
            output_dir: Directory to save processed crops
            
        Returns:
            Dictionary with processing statistics
        """
        stats = {'ev_cars': 0, 'non_ev_cars': 0, 'total_images': 0, 'failed_images': 0}
        
        for category in ['EV', 'Non-EV']:
            category_dir = os.path.join(dataset_dir, category)
            if not os.path.exists(category_dir):
                logger.warning(f"Directory not found: {category_dir}")
                continue
                
            output_category_dir = os.path.join(output_dir, category)
            os.makedirs(output_category_dir, exist_ok=True)
            
            image_files = [f for f in os.listdir(category_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            logger.info(f"Processing {len(image_files)} images in {category} category")
            
            for img_file in image_files:
                img_path = os.path.join(category_dir, img_file)
                stats['total_images'] += 1
                
                try:
                    crops, detections = self.process_image(
                        img_path, 
                        save_crops=True, 
                        output_dir=output_category_dir
                    )
                    
                    if category == 'EV':
                        stats['ev_cars'] += len(crops)
                    else:
                        stats['non_ev_cars'] += len(crops)
                        
                except Exception as e:
                    logger.error(f"Failed to process {img_path}: {str(e)}")
                    stats['failed_images'] += 1
        
        logger.info(f"Processing complete: {stats}")
        return stats
    
    def visualize_detections(self, image_path: str, output_path: str = None):
        """
        Visualize car detections on image
        
        Args:
            image_path: Path to input image
            output_path: Path to save visualization
        """
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return
        
        detections = self.detect_cars(image)
        
        # Draw bounding boxes
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class']
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, image)
        else:
            cv2.imshow('Car Detections', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    detector = CarDetector(model_name='yolov8s.pt', confidence_threshold=0.5)
    
    # Process dataset
    dataset_dir = "dataset"  # Should contain EV and Non-EV folders
    output_dir = "processed_dataset"
    
    if os.path.exists(dataset_dir):
        stats = detector.process_dataset(dataset_dir, output_dir)
        print(f"Processing completed: {stats}")
    else:
        print("Dataset directory not found. Please create 'dataset' folder with 'EV' and 'Non-EV' subfolders.")