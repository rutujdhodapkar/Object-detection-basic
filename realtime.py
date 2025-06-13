 import cv2
from ultralytics import YOLO
from flask import Flask, Response
import threading

app = Flask(__name__)

# Global camera object and lock for thread safety
camera = None
camera_lock = threading.Lock()

def gen_frames():
    global camera
    with camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(0)
    model = YOLO('yolov8s.pt')
    while True:
        with camera_lock:
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

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    # Simple HTML page to show the video stream
    return '''
    <html>
      <head>
        <title>YOLOv8 Real-time Object Detection</title>
      </head>
      <body>
        <h1>YOLOv8 Real-time Object Detection</h1>
        <img src="/video_feed" width="800" />
      </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
