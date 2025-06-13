from ultralytics import YOLO
import cv2

model = YOLO("yolov8s.pt")  # Corrected filename to lowercase, as per Ultralytics model naming

def detect_objects(image_path):
    results = model(image_path)
    result = results[0]
    # Save the image with detections
    result.save(filename='output_detected.jpg')
    # Get class names
    labels = result.names
    # Get detected class indices
    detected_classes = result.boxes.cls.cpu().numpy() if hasattr(result.boxes.cls, 'cpu') else result.boxes.cls
    detected = [labels[int(cls)] for cls in detected_classes]
    return list(set(detected)), 'output_detected.jpg'
