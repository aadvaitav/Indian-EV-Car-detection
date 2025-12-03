"""
Tkinter GUI Application for EV Car Detection and Classification
Provides user-friendly interface for image upload, camera feed, and batch processing
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import queue
import os
import json
from datetime import datetime
import logging

# Import custom modules
from inference import EVInference
from car_detector import CarDetector
from data_augmentation import DataAugmentor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EVDetectionApp:
    """Main GUI Application for EV Car Detection"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Indian EV Car Detection System")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.inference_engine = None
        self.model_loaded = False
        self.camera_active = False
        self.camera_thread = None
        self.current_image = None
        self.results_history = []
        
        # Camera variables
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=2)
        
        # Setup GUI
        self.setup_gui()
        
        # Try to load model automatically
        self.auto_load_model()
    
    def setup_gui(self):
        """Setup the main GUI layout"""
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_image_tab()
        self.create_camera_tab()
        self.create_batch_tab()
        self.create_results_tab()
        self.create_settings_tab()
        
        # Status bar
        self.create_status_bar()
    
    def create_image_tab(self):
        """Create image processing tab"""
        self.image_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.image_frame, text="Image Processing")
        
        # Left panel for controls
        left_panel = ttk.Frame(self.image_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Model status
        model_status_frame = ttk.LabelFrame(left_panel, text="Model Status", padding=10)
        model_status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.model_status_label = ttk.Label(model_status_frame, text="Model: Not Loaded", foreground="red")
        self.model_status_label.pack()
        
        ttk.Button(model_status_frame, text="Load Model", command=self.load_model).pack(pady=5)
        
        # Image upload
        upload_frame = ttk.LabelFrame(left_panel, text="Image Upload", padding=10)
        upload_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(upload_frame, text="Select Image", command=self.select_image).pack(fill=tk.X, pady=2)
        ttk.Button(upload_frame, text="Process Image", command=self.process_image).pack(fill=tk.X, pady=2)
        ttk.Button(upload_frame, text="Clear Image", command=self.clear_image).pack(fill=tk.X, pady=2)
        
        # Results display
        results_frame = ttk.LabelFrame(left_panel, text="Detection Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, width=30)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Right panel for image display
        right_panel = ttk.Frame(self.image_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Image display
        image_display_frame = ttk.LabelFrame(right_panel, text="Image Display", padding=10)
        image_display_frame.pack(fill=tk.BOTH, expand=True)
        
        self.image_canvas = tk.Canvas(image_display_frame, bg="white", width=600, height=500)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
    
    def create_camera_tab(self):
        """Create camera processing tab"""
        self.camera_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.camera_frame, text="Camera Feed")
        
        # Left panel for controls
        left_panel = ttk.Frame(self.camera_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Camera controls
        camera_controls = ttk.LabelFrame(left_panel, text="Camera Controls", padding=10)
        camera_controls.pack(fill=tk.X, pady=(0, 10))
        
        self.camera_status_label = ttk.Label(camera_controls, text="Camera: Inactive", foreground="red")
        self.camera_status_label.pack()
        
        ttk.Button(camera_controls, text="Start Camera", command=self.start_camera).pack(fill=tk.X, pady=2)
        ttk.Button(camera_controls, text="Stop Camera", command=self.stop_camera).pack(fill=tk.X, pady=2)
        ttk.Button(camera_controls, text="Capture Frame", command=self.capture_frame).pack(fill=tk.X, pady=2)
        
        # Camera settings
        settings_frame = ttk.LabelFrame(left_panel, text="Camera Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(settings_frame, text="Camera ID:").pack()
        self.camera_id_var = tk.StringVar(value="0")
        ttk.Entry(settings_frame, textvariable=self.camera_id_var).pack(fill=tk.X, pady=2)
        
        # Live statistics
        stats_frame = ttk.LabelFrame(left_panel, text="Live Statistics", padding=10)
        stats_frame.pack(fill=tk.BOTH, expand=True)
        
        self.stats_text = scrolledtext.ScrolledText(stats_frame, height=10, width=30)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
        # Right panel for camera display
        right_panel = ttk.Frame(self.camera_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Camera display
        camera_display_frame = ttk.LabelFrame(right_panel, text="Camera Feed", padding=10)
        camera_display_frame.pack(fill=tk.BOTH, expand=True)
        
        self.camera_canvas = tk.Canvas(camera_display_frame, bg="black", width=640, height=480)
        self.camera_canvas.pack(fill=tk.BOTH, expand=True)
    
    def create_batch_tab(self):
        """Create batch processing tab"""
        self.batch_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.batch_frame, text="Batch Processing")
        
        # Left panel for controls
        left_panel = ttk.Frame(self.batch_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Batch controls
        batch_controls = ttk.LabelFrame(left_panel, text="Batch Controls", padding=10)
        batch_controls.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(batch_controls, text="Select Folder", command=self.select_batch_folder).pack(fill=tk.X, pady=2)
        ttk.Button(batch_controls, text="Process Batch", command=self.process_batch).pack(fill=tk.X, pady=2)
        ttk.Button(batch_controls, text="Export Results", command=self.export_batch_results).pack(fill=tk.X, pady=2)
        
        # Progress bar
        self.batch_progress = ttk.Progressbar(batch_controls, mode='determinate')
        self.batch_progress.pack(fill=tk.X, pady=5)
        
        # Batch settings
        settings_frame = ttk.LabelFrame(left_panel, text="Batch Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.save_annotated_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Save Annotated Images", 
                       variable=self.save_annotated_var).pack(anchor=tk.W)
        
        self.detailed_report_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Generate Detailed Report", 
                       variable=self.detailed_report_var).pack(anchor=tk.W)
        
        # Right panel for batch results
        right_panel = ttk.Frame(self.batch_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Batch results
        batch_results_frame = ttk.LabelFrame(right_panel, text="Batch Results", padding=10)
        batch_results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.batch_text = scrolledtext.ScrolledText(batch_results_frame, height=25)
        self.batch_text.pack(fill=tk.BOTH, expand=True)
    
    def create_results_tab(self):
        """Create results history tab"""
        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text="Results History")
        
        # Results controls
        controls_frame = ttk.Frame(self.results_tab)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(controls_frame, text="Clear History", command=self.clear_history).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Export History", command=self.export_history).pack(side=tk.LEFT, padx=5)
        
        # Results treeview
        self.results_tree = ttk.Treeview(self.results_tab, columns=('Time', 'Type', 'Total Cars', 'EV Cars', 'Accuracy'), show='headings')
        
        # Define headings
        self.results_tree.heading('Time', text='Time')
        self.results_tree.heading('Type', text='Type')
        self.results_tree.heading('Total Cars', text='Total Cars')
        self.results_tree.heading('EV Cars', text='EV Cars')
        self.results_tree.heading('Accuracy', text='Confidence')
        
        # Configure column widths
        self.results_tree.column('Time', width=150)
        self.results_tree.column('Type', width=100)
        self.results_tree.column('Total Cars', width=100)
        self.results_tree.column('EV Cars', width=100)
        self.results_tree.column('Accuracy', width=100)
        
        # Pack treeview with scrollbar
        tree_frame = ttk.Frame(self.results_tab)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tree_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=tree_scrollbar.set)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_settings_tab(self):
        """Create settings tab"""
        self.settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_tab, text="Settings")
        
        # Model settings
        model_settings = ttk.LabelFrame(self.settings_tab, text="Model Settings", padding=10)
        model_settings.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(model_settings, text="Model Path:").pack(anchor=tk.W)
        self.model_path_var = tk.StringVar(value="best_model.pth")
        model_path_frame = ttk.Frame(model_settings)
        model_path_frame.pack(fill=tk.X, pady=5)
        ttk.Entry(model_path_frame, textvariable=self.model_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(model_path_frame, text="Browse", command=self.browse_model_path).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Detection settings
        detection_settings = ttk.LabelFrame(self.settings_tab, text="Detection Settings", padding=10)
        detection_settings.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(detection_settings, text="Confidence Threshold:").pack(anchor=tk.W)
        self.confidence_threshold_var = tk.DoubleVar(value=0.5)
        ttk.Scale(detection_settings, from_=0.1, to=1.0, variable=self.confidence_threshold_var,
                 orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)
        
        confidence_label = ttk.Label(detection_settings, text="")
        confidence_label.pack()
        
        def update_confidence_label(*args):
            confidence_label.config(text=f"Current: {self.confidence_threshold_var.get():.2f}")
        
        self.confidence_threshold_var.trace('w', update_confidence_label)
        update_confidence_label()
        
        # Performance settings
        performance_settings = ttk.LabelFrame(self.settings_tab, text="Performance Settings", padding=10)
        performance_settings.pack(fill=tk.X, padx=10, pady=10)
        
        # Performance settings
        performance_settings = ttk.LabelFrame(self.settings_tab, text="Performance Settings", padding=10)
        performance_settings.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(performance_settings, text="Device:").pack(anchor=tk.W)
        self.device_var = tk.StringVar(value="auto")
        device_combo = ttk.Combobox(performance_settings, textvariable=self.device_var, 
                                   values=["auto", "cpu", "cuda"], state="readonly")
        device_combo.pack(fill=tk.X, pady=2)
        
        # Application info
        info_frame = ttk.LabelFrame(self.settings_tab, text="Application Info", padding=10)
        info_frame.pack(fill=tk.X, padx=10, pady=10)
        
        info_text = """Indian EV Car Detection System
Version: 1.0
Author: AI Assistant
Description: Advanced CNN-based system for detecting and classifying electric vehicles in Indian traffic scenarios."""
        
        ttk.Label(info_frame, text=info_text, justify=tk.LEFT).pack()
    
    def create_status_bar(self):
        """Create status bar"""
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = ttk.Label(self.status_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Performance indicator
        self.performance_label = ttk.Label(self.status_frame, text="")
        self.performance_label.pack(side=tk.RIGHT, padx=10)
    
    def update_status(self, message):
        """Update status bar message"""
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def auto_load_model(self):
        """Automatically load model if available"""
        model_path = self.model_path_var.get()
        if os.path.exists(model_path):
            self.load_model()
    
    def load_model(self):
        """Load the EV classification model"""
        model_path = self.model_path_var.get()
        
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Model file not found: {model_path}")
            return
        
        try:
            self.update_status("Loading model...")
            self.inference_engine = EVInference(model_path, device=self.device_var.get())
            self.model_loaded = True
            self.model_status_label.config(text="Model: Loaded ✓", foreground="green")
            self.update_status("Model loaded successfully")
            messagebox.showinfo("Success", "Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.update_status("Model loading failed")
    
    def browse_model_path(self):
        """Browse for model file"""
        filename = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")]
        )
        if filename:
            self.model_path_var.set(filename)
    
    def select_image(self):
        """Select image for processing"""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Please load the model first!")
            return
        
        filename = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")]
        )
        
        if filename:
            try:
                # Load and display image
                self.current_image = cv2.imread(filename)
                self.display_image(self.current_image)
                self.update_status(f"Image loaded: {os.path.basename(filename)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def display_image(self, image, canvas=None):
        """Display image on canvas"""
        if canvas is None:
            canvas = self.image_canvas
        
        # Convert BGR to RGB
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Resize image to fit canvas
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not yet initialized, use default size
            canvas_width, canvas_height = 600, 500
        
        h, w = image_rgb.shape[:2]
        scale = min(canvas_width / w, canvas_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized_image = cv2.resize(image_rgb, (new_w, new_h))
        
        # Convert to PIL Image and then to PhotoImage
        pil_image = Image.fromarray(resized_image)
        photo = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and display image
        canvas.delete("all")
        canvas.create_image(canvas_width//2, canvas_height//2, anchor=tk.CENTER, image=photo)
        canvas.image = photo  # Keep a reference
    
    def process_image(self):
        """Process current image for EV detection"""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Please load the model first!")
            return
        
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please select an image first!")
            return
        
        try:
            self.update_status("Processing image...")
            
            # Get prediction results
            results = self.inference_engine.predict_with_detection(self.current_image)
            
            # Display annotated image
            if results:
                annotated_image = self.inference_engine._annotate_frame(self.current_image, results)
                self.display_image(annotated_image)
            
            # Update results text
            self.display_image_results(results)
            
            # Add to history
            self.add_to_history("Image", results)
            
            self.update_status("Image processed successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
            self.update_status("Image processing failed")
    
    def display_image_results(self, results):
        """Display image processing results"""
        self.results_text.delete(1.0, tk.END)
        
        if not results:
            self.results_text.insert(tk.END, "No cars detected in the image.\n")
            return
        
        ev_count = sum(1 for r in results if r['is_ev'])
        total_cars = len(results)
        
        summary = f"DETECTION SUMMARY\n"
        summary += f"="*30 + "\n"
        summary += f"Total Cars: {total_cars}\n"
        summary += f"EV Cars: {ev_count}\n"
        summary += f"Non-EV Cars: {total_cars - ev_count}\n"
        summary += f"EV Percentage: {(ev_count/total_cars*100):.1f}%\n\n"
        
        summary += f"DETAILED RESULTS\n"
        summary += f"="*30 + "\n"
        
        for i, result in enumerate(results, 1):
            summary += f"Car {i}:\n"
            summary += f"  Type: {result['ev_prediction']}\n"
            summary += f"  Confidence: {result['ev_confidence']:.3f}\n"
            summary += f"  Detection Confidence: {result['car_confidence']:.3f}\n"
            summary += f"  Bbox: {result['bbox']}\n\n"
        
        self.results_text.insert(tk.END, summary)
    
    def clear_image(self):
        """Clear current image"""
        self.current_image = None
        self.image_canvas.delete("all")
        self.results_text.delete(1.0, tk.END)
        self.update_status("Image cleared")
    
    def start_camera(self):
        """Start camera feed"""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Please load the model first!")
            return
        
        if self.camera_active:
            return
        
        try:
            camera_id = int(self.camera_id_var.get())
            self.cap = cv2.VideoCapture(camera_id)
            
            if not self.cap.isOpened():
                messagebox.showerror("Error", f"Could not open camera {camera_id}")
                return
            
            self.camera_active = True
            self.camera_status_label.config(text="Camera: Active ✓", foreground="green")
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
            # Start display update
            self.update_camera_display()
            
            self.update_status("Camera started")
            
        except ValueError:
            messagebox.showerror("Error", "Invalid camera ID")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
    
    def stop_camera(self):
        """Stop camera feed"""
        if not self.camera_active:
            return
        
        self.camera_active = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.camera_status_label.config(text="Camera: Inactive", foreground="red")
        self.camera_canvas.delete("all")
        self.update_status("Camera stopped")
    
    def camera_loop(self):
        """Camera capture loop (runs in separate thread)"""
        frame_count = 0
        total_evs = 0
        
        while self.camera_active:
            if self.cap:
                ret, frame = self.cap.read()
                if ret:
                    # Process frame
                    annotated_frame, results = self.inference_engine.process_video_frame(frame)
                    
                    # Update statistics
                    frame_count += 1
                    total_evs += results['ev_cars']
                    
                    # Put frame in queue for display
                    if not self.frame_queue.full():
                        try:
                            self.frame_queue.put_nowait((annotated_frame, results, frame_count, total_evs))
                        except queue.Full:
                            pass
                
            # Small delay to prevent excessive CPU usage
            threading.Event().wait(0.03)  # ~30 FPS
    
    def update_camera_display(self):
        """Update camera display (runs in main thread)"""
        if not self.camera_active:
            return
        
        try:
            # Get latest frame from queue
            frame, results, frame_count, total_evs = self.frame_queue.get_nowait()
            
            # Display frame
            self.display_image(frame, self.camera_canvas)
            
            # Update statistics
            avg_evs = total_evs / frame_count if frame_count > 0 else 0
            stats_text = f"Frame: {frame_count}\n"
            stats_text += f"Total EVs: {total_evs}\n"
            stats_text += f"Avg EVs/frame: {avg_evs:.2f}\n"
            stats_text += f"Current Cars: {results['total_cars']}\n"
            stats_text += f"Current EVs: {results['ev_cars']}\n"
            
            # Performance stats
            if self.inference_engine:
                perf_stats = self.inference_engine.get_performance_stats()
                if 'avg_fps' in perf_stats:
                    stats_text += f"FPS: {perf_stats['avg_fps']:.1f}\n"
            
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, stats_text)
            
        except queue.Empty:
            pass
        
        # Schedule next update
        if self.camera_active:
            self.root.after(100, self.update_camera_display)
    
    def capture_frame(self):
        """Capture current camera frame"""
        if not self.camera_active or not self.cap:
            messagebox.showwarning("Warning", "Camera is not active!")
            return
        
        ret, frame = self.cap.read()
        if ret:
            # Save frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captured_frame_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            
            # Process frame
            results = self.inference_engine.predict_with_detection(frame)
            self.add_to_history("Camera Capture", results)
            
            messagebox.showinfo("Success", f"Frame captured and saved as {filename}")
    
    def select_batch_folder(self):
        """Select folder for batch processing"""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Please load the model first!")
            return
        
        folder_path = filedialog.askdirectory(title="Select Folder with Images")
        if folder_path:
            self.batch_folder_path = folder_path
            self.batch_text.delete(1.0, tk.END)
            self.batch_text.insert(tk.END, f"Selected folder: {folder_path}\n\n")
            
            # Count images in folder
            image_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            self.batch_text.insert(tk.END, f"Found {len(image_files)} images\n")
    
    def process_batch(self):
        """Process batch of images"""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Please load the model first!")
            return
        
        if not hasattr(self, 'batch_folder_path'):
            messagebox.showwarning("Warning", "Please select a folder first!")
            return
        
        # Run batch processing in separate thread
        self.batch_thread = threading.Thread(target=self.batch_processing_worker, daemon=True)
        self.batch_thread.start()
    
    def batch_processing_worker(self):
        """Batch processing worker thread"""
        try:
            folder_path = self.batch_folder_path
            
            # Get all image files
            image_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            total_files = len(image_files)
            if total_files == 0:
                self.root.after(0, lambda: messagebox.showinfo("Info", "No images found in selected folder"))
                return
            
            # Initialize progress
            self.root.after(0, lambda: self.batch_progress.config(maximum=total_files, value=0))
            
            # Process each image
            batch_results = []
            total_cars = 0
            total_evs = 0
            
            for i, filename in enumerate(image_files):
                image_path = os.path.join(folder_path, filename)
                
                try:
                    # Process image
                    result = self.inference_engine.predict_image_file(image_path)
                    batch_results.append(result)
                    
                    total_cars += result['total_cars_detected']
                    total_evs += result['ev_cars_detected']
                    
                    # Update progress
                    progress_text = f"Processing: {filename} ({i+1}/{total_files})\n"
                    progress_text += f"Cars: {result['total_cars_detected']}, EVs: {result['ev_cars_detected']}\n\n"
                    
                    self.root.after(0, lambda text=progress_text: self.batch_text.insert(tk.END, text))
                    self.root.after(0, lambda: self.batch_progress.config(value=i+1))
                    
                except Exception as e:
                    error_text = f"Error processing {filename}: {str(e)}\n\n"
                    self.root.after(0, lambda text=error_text: self.batch_text.insert(tk.END, text))
            
            # Final summary
            summary = f"\nBATCH PROCESSING COMPLETE\n"
            summary += f"=" * 40 + "\n"
            summary += f"Total Images Processed: {len(batch_results)}\n"
            summary += f"Total Cars Detected: {total_cars}\n"
            summary += f"Total EV Cars: {total_evs}\n"
            summary += f"Total Non-EV Cars: {total_cars - total_evs}\n"
            summary += f"EV Percentage: {(total_evs/total_cars*100):.1f}%\n" if total_cars > 0 else "EV Percentage: 0%\n"
            
            self.root.after(0, lambda: self.batch_text.insert(tk.END, summary))
            self.batch_results = batch_results
            
        except Exception as e:
            error_msg = f"Batch processing error: {str(e)}"
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
    
    def export_batch_results(self):
        """Export batch processing results"""
        if not hasattr(self, 'batch_results'):
            messagebox.showwarning("Warning", "No batch results to export!")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Batch Results",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.batch_results, f, indent=2)
                messagebox.showinfo("Success", f"Batch results exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {str(e)}")
    
    def add_to_history(self, process_type, results):
        """Add results to history"""
        if not results:
            return
        
        ev_count = sum(1 for r in results if r['is_ev'])
        total_cars = len(results)
        avg_confidence = np.mean([r['ev_confidence'] for r in results]) if results else 0
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add to treeview
        self.results_tree.insert('', 0, values=(
            timestamp,
            process_type,
            total_cars,
            ev_count,
            f"{avg_confidence:.3f}"
        ))
        
        # Add to internal history
        history_entry = {
            'timestamp': timestamp,
            'type': process_type,
            'total_cars': total_cars,
            'ev_cars': ev_count,
            'results': results
        }
        self.results_history.append(history_entry)
    
    def clear_history(self):
        """Clear results history"""
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.results_history.clear()
        messagebox.showinfo("Info", "History cleared successfully")
    
    def export_history(self):
        """Export results history"""
        if not self.results_history:
            messagebox.showwarning("Warning", "No history to export!")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save History",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.results_history, f, indent=2)
                messagebox.showinfo("Success", f"History exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export history: {str(e)}")

def main():
    """Main application entry point"""
    root = tk.Tk()
    app = EVDetectionApp(root)
    
    def on_closing():
        """Handle application closing"""
        if app.camera_active:
            app.stop_camera()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()