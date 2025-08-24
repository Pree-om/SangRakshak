  # Weapon Detection using YOLOv8

This project is a deep learning-based Weapon Detection Model built with the Ultralytics YOLOv8
 framework.
It detects and classifies different types of weapons (e.g., rifles, handguns, knives, etc.) in images, videos, and live streams.

## Features

**Real-time weapon detection on images, videos, and webcam streams**

**Training & validation metrics with confusion matrix visualization**

**Model trained on a custom dataset**

**Easy to run inference with best.pt**

**Results stored automatically in the runs/ folder**

## Tech Stack

Framework: Ultralytics YOLOv8

Language: Python

Libraries:

ultralytics

torch

opencv-python

matplotlib, pandas, seaborn (for visualization)


## Dataset

Download Dataset: 

Organized as:

dataset/
  - train/
    images/
labels/
  - Val/
 images/
     labels/

 **Training**
yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=50 imgsz=640

**Validation**
yolo val model=runs/detect/train/weights/best.pt data=dataset/data.yaml plots=True


Generates:

confusion_matrix.png

results.png

 Inference
# Image Prediction
yolo predict model=runs/detect/train/weights/best.pt source=path/to/image.jpg

#  GIF Prediction
yolo predict model=runs/detect/train/weights/best.pt source=path/to/animation.gif


# Webcam Stream
yolo predict model=runs/detect/train/weights/best.pt source=0


Results saved in runs/detect/predict/.

**Visualization**
Training Metrics
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("runs/detect/train/results.png")
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis("off")
plt.show()

Confusion Matrix
img = Image.open("runs/detect/train/confusion_matrix.png")
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis("off")
plt.show()

 Project Structure
weapon-detection-yolov8/
 dataset/
  runs/
      detect/
            train/
               weights/
                best.pt
               results.png
 confusion_matrix.png
 README.md
 


## Author
 Name: **Smriti Pandey**
GitHub: https://github.com/student-smritipandey