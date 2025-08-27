# YOLOv5 Real-Time Object Detection Desktop Application 🎥🖥️

This project is a **real-time object detection desktop application** that integrates the **YOLOv5 deep learning model** with a Python **Tkinter GUI**.  
It enables live detection of multiple objects from a webcam feed, providing bounding boxes, labels, and confidence scores.

---

## 🚀 Features
- **Real-Time Multi-Class Detection** using YOLOv5s (pre-trained on COCO dataset, 80 classes).
- **Adjustable Confidence Threshold** via GUI slider.
- **Interactive Desktop Application** built with Tkinter.
- **Threaded Processing** for smooth UI performance.
- **Color-Coded Visualization** for key objects (person, car, dog, cat).
- **Detection Summary Panel** showing objects counted per frame.

---

## 🧠 Technology Overview

- **YOLOv5** – Single-stage object detector (fast + accurate).
- **PyTorch** – Deep learning backend for inference.
- **OpenCV** – Webcam capture & visualization.
- **Tkinter** – GUI for user interaction.
- **PIL (Pillow)** – Image handling & Tkinter display.
- **Threading** – Ensures real-time processing without freezing the GUI.

---

## 🖥️ System Architecture

1. **Webcam Feed** captured via OpenCV.  
2. **YOLOv5 Model** processes each frame → bounding boxes + class labels + confidence.  
3. **Non-Max Suppression (NMS)** removes duplicate detections.  
4. **Bounding Boxes & Labels** drawn on the frame.  
5. **Tkinter GUI** displays results in real-time.  
6. **User Controls**: Start/Stop camera, adjust detection confidence.  

---
## ⚙️ Requirements

The project requires the following main libraries:

- **torch**  
- **torchvision**  
- **opencv-python**  
- **numpy**  
- **pillow**  
- **tkinter** (comes pre-installed with Python)
## ▶️ Usage

Run the app with:

    python app.py

### GUI Controls:
    Start Camera  → Starts webcam feed + object detection
    Stop Camera   → Stops webcam & detection
    Confidence Slider → Adjusts detection sensitivity
    Status Bar    → Displays detection and system messages

---

## 🎯 Example Output

    - Bounding boxes drawn on detected objects
    - Objects like person, car, dog, cat are highlighted with different colors
    - Bottom info panel shows counts, e.g.:

    Objects detected: 2 person, 1 car

---

## 🛠 Applications

    - Security & surveillance systems
    - Smart homes & IoT vision systems
    - Educational AI/computer vision demos
    - Prototype for vision-based automation

---

## 📌 Extensibility

    - Replace yolov5s with larger YOLOv5 variants for higher accuracy
    - Add video recording or multiple camera support
    - Integrate with cloud/IoT for remote monitoring
