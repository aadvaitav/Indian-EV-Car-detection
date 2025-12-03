"""
Inference Module for EV Car Classification
Handles single image prediction, batch processing, and live camera inference
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import os
import time
from typing import List, Tuple, Union, Optional
import logging
from collections import deque

# Import custom modules
from ev_classifier import create_model
from car_detector import CarDetector
from data_augmentation import DataAugmentor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EVInference:
    """EV Car Classification Inference Engine"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to saved model checkpoint
            device: Device to run inference on ('auto', 'cpu', or 'cuda')
        """
        self.device = self._setup_device(device)
        self.model = self._load_model(model_path)
        
        # Initialize car detector
        self.car_detector = CarDetector(model_name='yolov8s.pt', confidence_threshold=0.5)
        
        # Initialize data augmentor for preprocessing
        self.augmentor = DataAugmentor(target_size=(224, 224))
        self.transform = self.augmentor.get_transforms('inference')
        
        # Class labels
        self.class_names = ['Non-EV', 'EV']
        
        # Performance tracking
        self.inference_times = deque(maxlen=100)
        
        logger.info(f"Inference engine initialized on {self.device}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            return torch.device(device)
    
    def _load_model(self, model_path: str):
        """Load trained model from checkpoint"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model with saved configuration
        model_config = checkpoint.get('config', {}).get('model', {})
        if not model_config:
            # Default configuration if not found
            model_config = {
                'model_name': 'efficientnet_b3',
                'num_classes': 2,
                'dropout_rate': 0.3,
                'use_attention': True
            }
        
        model = create_model(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        logger.info(f"Model loaded from {model_path}")
        return model
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        transformed = self.transform(image=image)
        tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
        
        return tensor.to(self.device)
    
    def predict_single(self, image: np.ndarray, return_confidence: bool = True) -> Union[str, Tuple[str, float]]:
        """
        Predict EV classification for single image
        
        Args:
            image: Input image as numpy array
            return_confidence: Whether to return confidence score
            
        Returns:
            Prediction label or (label, confidence) tuple
        """
        start_time = time.time()
        
        # Preprocess image
        input_tensor = self.preprocess_image(image)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            prediction = self.class_names[predicted.item()]
            conf_score = confidence.item()
        
        # Track inference time
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        if return_confidence:
            return prediction, conf_score
        else:
            return prediction
    
    def predict_with_detection(self, image: np.ndarray) -> List[dict]:
        """
        Detect cars and classify each detected car as EV or Non-EV
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detection results with EV classification
        """
        # Detect cars
        detections = self.car_detector.detect_cars(image)
        
        if not detections:
            return []
        
        # Extract car crops
        car_crops = self.car_detector.extract_car_crops(image, detections)
        
        # Classify each car
        results = []
        for i, (detection, crop) in enumerate(zip(detections, car_crops)):
            if crop.size > 0:
                prediction, confidence = self.predict_single(crop, return_confidence=True)
                
                result = {
                    'detection_id': i,
                    'bbox': detection['bbox'],
                    'car_confidence': detection['confidence'],
                    'car_class': detection['class'],
                    'ev_prediction': prediction,
                    'ev_confidence': confidence,
                    'is_ev': prediction == 'EV'
                }
                results.append(result)
        
        return results
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Predict EV classification for batch of images
        
        Args:
            images: List of input images as numpy arrays
            
        Returns:
            List of (prediction, confidence) tuples
        """
        if not images:
            return []
        
        # Preprocess all images
        batch_tensors = []
        for image in images:
            tensor = self.preprocess_image(image)
            batch_tensors.append(tensor.squeeze(0))  # Remove batch dimension
        
        # Create batch tensor
        batch_tensor = torch.stack(batch_tensors).to(self.device)
        
        # Batch inference
        results = []
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, 1)
            
            for pred, conf in zip(predictions, confidences):
                prediction = self.class_names[pred.item()]
                confidence = conf.item()
                results.append((prediction, confidence))
        
        return results
    
    def predict_image_file(self, image_path: str) -> dict:
        """
        Predict EV classification from image file
        
        Args:
            image_path: Path to input image file
            
        Returns:
            Dictionary with prediction results
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Get predictions with car detection
        results = self.predict_with_detection(image)
        
        # Count EVs
        ev_count = sum(1 for result in results if result['is_ev'])
        total_cars = len(results)
        
        return {
            'image_path': image_path,
            'total_cars_detected': total_cars,
            'ev_cars_detected': ev_count,
            'non_ev_cars_detected': total_cars - ev_count,
            'detection_results': results
        }
    
    def process_video_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Process single video frame for EV detection and classification
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (annotated_frame, results_dict)
        """
        # Get predictions
        results = self.predict_with_detection(frame)
        
        # Count EVs
        ev_count = sum(1 for result in results if result['is_ev'])
        total_cars = len(results)
        
        # Annotate frame
        annotated_frame = self._annotate_frame(frame, results)
        
        # Add summary text
        summary_text = f"Cars: {total_cars} | EVs: {ev_count} | Non-EVs: {total_cars - ev_count}"
        cv2.putText(annotated_frame, summary_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add FPS info
        if self.inference_times:
            avg_time = np.mean(self.inference_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(annotated_frame, fps_text, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        results_dict = {
            'total_cars': total_cars,
            'ev_cars': ev_count,
            'non_ev_cars': total_cars - ev_count,
            'detections': results
        }
        
        return annotated_frame, results_dict
    
    def _annotate_frame(self, frame: np.ndarray, results: List[dict]) -> np.ndarray:
        """
        Annotate frame with detection and classification results
        
        Args:
            frame: Input frame
            results: Detection results
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            is_ev = result['is_ev']
            ev_confidence = result['ev_confidence']
            
            # Choose color based on EV classification
            color = (0, 255, 0) if is_ev else (0, 0, 255)  # Green for EV, Red for Non-EV
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"{result['ev_prediction']}: {ev_confidence:.2f}"
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame
    
    def process_live_camera(self, camera_id: int = 0, display: bool = True, save_video: bool = False):
        """
        Process live camera feed for real-time EV detection
        
        Args:
            camera_id: Camera device ID
            display: Whether to display the video feed
            save_video: Whether to save the output video
        """
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logger.error(f"Could not open camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Video writer setup
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter('output_camera.mp4', fourcc, 20.0, (1280, 720))
        
        logger.info("Starting live camera processing. Press 'q' to quit.")
        
        frame_count = 0
        total_ev_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process frame
                annotated_frame, results = self.process_video_frame(frame)
                total_ev_count += results['ev_cars']
                
                # Save video frame
                if save_video and video_writer:
                    video_writer.write(annotated_frame)
                
                # Display frame
                if display:
                    cv2.imshow('EV Car Detection', annotated_frame)
                    
                    # Check for quit key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Log periodic stats
                if frame_count % 100 == 0:
                    avg_ev_per_frame = total_ev_count / frame_count
                    logger.info(f"Processed {frame_count} frames, Average EVs per frame: {avg_ev_per_frame:.2f}")
        
        except KeyboardInterrupt:
            logger.info("Camera processing interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if video_writer:
                video_writer.release()
            if display:
                cv2.destroyAllWindows()
            
            logger.info(f"Camera processing completed. Total frames: {frame_count}, Total EVs detected: {total_ev_count}")
    
    def process_video_file(self, video_path: str, output_path: str = None, skip_frames: int = 1):
        """
        Process video file for EV detection and classification
        
        Args:
            video_path: Path to input video file
            output_path: Path to save output video (optional)
            skip_frames: Process every nth frame for faster processing
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Total frames: {total_frames}, FPS: {fps}")
        
        # Video writer
        video_writer = None
        if output_path:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        processed_count = 0
        total_ev_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames for faster processing
                if frame_count % skip_frames != 0:
                    if video_writer:
                        video_writer.write(frame)  # Write original frame
                    continue
                
                # Process frame
                annotated_frame, results = self.process_video_frame(frame)
                total_ev_count += results['ev_cars']
                processed_count += 1
                
                # Save processed frame
                if video_writer:
                    video_writer.write(annotated_frame)
                
                # Progress update
                if processed_count % 50 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}% - EVs detected: {total_ev_count}")
        
        finally:
            # Cleanup
            cap.release()
            if video_writer:
                video_writer.release()
            
            avg_ev_per_frame = total_ev_count / processed_count if processed_count > 0 else 0
            logger.info(f"Video processing completed!")
            logger.info(f"Processed {processed_count} frames, Total EVs: {total_ev_count}")
            logger.info(f"Average EVs per processed frame: {avg_ev_per_frame:.2f}")
    
    def get_performance_stats(self) -> dict:
        """Get inference performance statistics"""
        if not self.inference_times:
            return {'message': 'No inference performed yet'}
        
        times = np.array(self.inference_times)
        return {
            'avg_inference_time': float(np.mean(times)),
            'min_inference_time': float(np.min(times)),
            'max_inference_time': float(np.max(times)),
            'avg_fps': float(1.0 / np.mean(times)),
            'total_inferences': len(times)
        }

def main():
    """Example usage of the inference engine"""
    model_path = 'best_model.pth'
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        logger.info("Please train the model first using main.py")
        return
    
    # Initialize inference engine
    inference_engine = EVInference(model_path)
    
    # Example 1: Process single image
    test_image_path = 'test_image.jpg'
    if os.path.exists(test_image_path):
        result = inference_engine.predict_image_file(test_image_path)
        logger.info(f"Image prediction result: {result}")
    
    # Example 2: Process live camera (uncomment to use)
    # inference_engine.process_live_camera(camera_id=0, display=True, save_video=False)
    
    # Example 3: Process video file (uncomment to use)
    test_video_path = r'C:\Users\aadva\Downloads\test_images\no_ev_video.mp4'
    if os.path.exists(test_video_path):
        inference_engine.process_video_file(test_video_path, 'output_video.mp4')
    
    # Show performance stats
    stats = inference_engine.get_performance_stats()
    logger.info(f"Performance stats: {stats}")

if __name__ == "__main__":
    main()