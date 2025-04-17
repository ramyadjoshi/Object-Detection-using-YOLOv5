# Object-Detection-using-YOLOv5
Developed a project which focuses on implementing real-time object detection using the YOLOv5 framework. 
The primary goal is to detect and classify objects in video streams with high accuracy and efficiency.

# Import the requirements 
import tkinter as tk
from tkinter import messagebox
import cv2
import torch
from PIL import Image, ImageTk

class ObjectDetectionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Real-Time Object Detection with YOLOv5")
        
        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, autoshape=True)
        
        # Create a label to display the video feed
        self.video_label = tk.Label(master)
        self.video_label.pack()

        # Create a start button
        self.start_button = tk.Button(master, text="Start Detection", command=self.start_detection)
        self.start_button.pack()

        # Create a stop button
        self.stop_button = tk.Button(master, text="Stop Detection", command=self.stop_detection)
        self.stop_button.pack()

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.running = False

    def start_detection(self):
        if not self.running:
            self.running = True
            self.update_frame()

    def stop_detection(self):
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(rgb_frame)
            frame_with_detections = results.render()[0]
            frame_with_detections = cv2.cvtColor(frame_with_detections, cv2.COLOR_RGB2BGR)

            # Convert image to PhotoImage for tkinter
            img = Image.fromarray(frame_with_detections)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            # Schedule the next frame update
            self.video_label.after(10, self.update_frame)

# Create the main window
root = tk.Tk()
app = ObjectDetectionApp(root)

# Start the GUI event loop
root.mainloop()
