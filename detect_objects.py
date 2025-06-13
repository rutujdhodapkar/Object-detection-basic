from ultralytics import YOLO
import cv2

model = YOLO("yolov8x.pt")  # big boi version

def detect_objects(image_path):
    results = model(image_path)
    results[0].save(filename='output_detected.jpg')
    labels = results[0].names
    boxes = results[0].boxes
    detected = [labels[int(cls)] for cls in boxes.cls]
    return list(set(detected)), 'output_detected.jpg'
