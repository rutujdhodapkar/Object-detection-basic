from ultralytics import YOLO
import cv2
import time
from flask import Response

model = YOLO("yolov8s.pt")  # Corrected filename to lowercase, as per Ultralytics model naming

def detect_objects(image_path=None, use_camera=False):
    """
    Detect objects in an image or from live camera.
    If use_camera is True, captures an image from webcam every 0.5 seconds, predicts, and displays.
    Press 'q' to quit. The last detected frame is saved as 'output_detected.jpg'.
    If image_path is provided and use_camera is False, detects objects in the image.
    Returns (detected_objects, output_image_path) for image mode,
    or (detected_objects, output_image_path) for camera mode (last frame).
    """
    if use_camera:
        cap = cv2.VideoCapture(0)
        detected = []
        output_path = 'output_detected.jpg'
        last_capture_time = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            current_time = time.time()
            # Capture and predict every 0.5 seconds
            if current_time - last_capture_time >= 0.5:
                results = model(frame)
                result = results[0]
                # Draw detections on frame
                annotated_frame = result.plot()
                # Get class names
                labels = result.names
                detected_classes = result.boxes.cls.cpu().numpy() if hasattr(result.boxes.cls, 'cpu') else result.boxes.cls
                detected = [labels[int(cls)] for cls in detected_classes]
                # Show the frame
                cv2.imshow("Live Object Detection - Press 'q' to quit", annotated_frame)
                # Save the last frame with detections
                cv2.imwrite(output_path, annotated_frame)
                last_capture_time = current_time
            else:
                # Show the last frame if not time to capture
                cv2.imshow("Live Object Detection - Press 'q' to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        return list(set(detected)), output_path
    else:
        results = model(image_path)
        result = results[0]
        # Save the image with detections
        result.save(filename='output_detected.jpg')
        # Get class names
        labels = result.names
        detected_classes = result.boxes.cls.cpu().numpy() if hasattr(result.boxes.cls, 'cpu') else result.boxes.cls
        detected = [labels[int(cls)] for cls in detected_classes]
        return list(set(detected)), 'output_detected.jpg'

# --- Real-time YOLOv8 camera stream algorithm for Flask (for use in app.py or similar) ---
def gen_frames_realtime():
    """
    Generator function for real-time YOLOv8 detection streaming (Flask Response).
    Yields JPEG-encoded frames with detection boxes, suitable for multipart/x-mixed-replace.
    """
    camera = cv2.VideoCapture(0)
    model = YOLO('yolov8s.pt')
    while True:
        success, frame = camera.read()
        if not success:
            break
        # Run YOLOv8 object detection on the frame
        results = model(frame)
        # Draw detection results on the frame
        annotated_frame = results[0].plot()
        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        # Yield frame in multipart/x-mixed-replace format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    camera.release()
