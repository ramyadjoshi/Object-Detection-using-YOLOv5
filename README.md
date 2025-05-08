import torch
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import time

class ObjectDetectionApp:
    def __init__(self, window, window_title):
        # Initialize the main window
        self.window = window
        self.window.title(window_title)
        self.window.geometry("800x600")
        self.window.resizable(width=True, height=True)
        
        # Load the YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        
        # Variables
        self.camera_active = False
        self.cap = None
        self.detection_thread = None
        self.stop_thread = False
        
        # Create GUI elements
        self.create_widgets()
        
        # Protocol for closing the window
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def create_widgets(self):
        # Top frame for controls
        control_frame = ttk.Frame(self.window)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        # Start camera button
        self.start_button = ttk.Button(control_frame, text="Start Camera", command=self.start_camera)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # Stop camera button
        self.stop_button = ttk.Button(control_frame, text="Stop Camera", command=self.stop_camera, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Confidence threshold slider
        self.conf_threshold = tk.DoubleVar(value=0.3)
        ttk.Label(control_frame, text="Confidence Threshold:").pack(side=tk.LEFT, padx=(20, 5))
        threshold_slider = ttk.Scale(control_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL, 
                                    variable=self.conf_threshold, length=200)
        threshold_slider.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(control_frame, textvariable=self.conf_threshold).pack(side=tk.LEFT, padx=5)
        
        # Frame for video display
        self.display_frame = ttk.Frame(self.window, borderwidth=2, relief="sunken")
        self.display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Label for displaying video feed
        self.video_label = ttk.Label(self.display_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Status: Ready")
        status_bar = ttk.Label(self.window, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Information about detection
        self.info_frame = ttk.Frame(self.window)
        self.info_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        self.detection_info = tk.StringVar(value="Objects detected: None")
        ttk.Label(self.info_frame, textvariable=self.detection_info).pack(side=tk.LEFT)
    
    def start_camera(self):
        if not self.camera_active:
            try:
                self.cap = cv2.VideoCapture(0)  # Open webcam
                if not self.cap.isOpened():
                    messagebox.showerror("Error", "Could not open webcam!")
                    return
                
                self.camera_active = True
                self.stop_thread = False
                
                # Update UI
                self.start_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.NORMAL)
                self.status_var.set("Status: Camera active - Detecting objects")
                
                # Start detection in a separate thread
                self.detection_thread = threading.Thread(target=self.detection_loop)
                self.detection_thread.daemon = True
                self.detection_thread.start()
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not start detection: {str(e)}")
    
    def stop_camera(self):
        if self.camera_active:
            self.stop_thread = True
            self.camera_active = False
            
            # Update UI
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_var.set("Status: Camera stopped")
            
            # Release the camera
            if self.cap is not None:
                self.cap.release()
                self.cap = None
    
    def detect_objects(self, image):
        # Perform inference on the image
        results = self.model(image)
        # Extract detected objects
        labels, cord = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()
        return labels, cord, results
    
    def plot_boxes(self, frame, labels, cord):
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        confidence_threshold = self.conf_threshold.get()
        
        # Count objects above threshold
        detected_objects = {}
        
        for i in range(n):
            row = cord[i]
            if row[4] >= confidence_threshold:  # Confidence threshold
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
                label = int(labels[i])
                class_name = self.model.names[label]
                
                # Update object counter
                if class_name in detected_objects:
                    detected_objects[class_name] += 1
                else:
                    detected_objects[class_name] = 1
                
                # Different colors for different classes
                color_mapping = {
                    'person': (0, 255, 0),  # Green
                    'car': (255, 0, 0),     # Blue
                    'dog': (0, 0, 255),     # Red
                    'cat': (255, 255, 0),   # Cyan
                }
                
                bgr = color_mapping.get(class_name, (0, 255, 0))  # Default to green
                
                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                
                # Add label and confidence score
                conf_text = f"{class_name}: {row[4]:.2f}"
                cv2.putText(frame, conf_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
        
        # Update detection info
        if detected_objects:
            info_text = "Objects detected: " + ", ".join([f"{count} {name}" for name, count in detected_objects.items()])
            self.detection_info.set(info_text)
        else:
            self.detection_info.set("Objects detected: None")
        
        return frame
    
    def detection_loop(self):
        try:
            while not self.stop_thread:
                ret, frame = self.cap.read()
                
                if not ret:
                    self.status_var.set("Status: Error reading from camera")
                    break
                
                # Object detection
                labels, cord, results = self.detect_objects(frame)
                frame = self.plot_boxes(frame, labels, cord)
                
                # Convert to format suitable for Tkinter
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                
                # Resize to fit the display
                display_width = self.display_frame.winfo_width()
                display_height = self.display_frame.winfo_height()
                
                # Make sure we have valid dimensions
                if display_width > 10 and display_height > 10:
                    img = img.resize((display_width, display_height), Image.LANCZOS)
                
                imgtk = ImageTk.PhotoImage(image=img)
                
                # Update the video label
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                
                # Sleep briefly to reduce CPU usage
                time.sleep(0.01)
                
        except Exception as e:
            self.status_var.set(f"Status: Error in detection loop - {str(e)}")
            messagebox.showerror("Error", f"Detection error: {str(e)}")
        finally:
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
    
    def on_closing(self):
        self.stop_camera()
        self.window.destroy()


if __name__ == "__main__":
    # Create the main window
    root = tk.Tk()
    app = ObjectDetectionApp(root, "YOLOv5 Real-Time Object Detection")
    
    # Start the Tkinter event loop
    root.mainloop()
