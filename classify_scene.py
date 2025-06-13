import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import time
from flask import Flask, Response
import threading

app = Flask(__name__)

# Global camera object and lock for thread safety (from realtime.py)
camera = None
camera_lock = threading.Lock()

def load_places365():
    model = models.resnet50(num_classes=365)
    checkpoint = torch.hub.load_state_dict_from_url(
        'http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar', map_location='cpu'
    )
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model

def get_places365_classes():
    with open('categories_places365.txt') as f:
        classes = [line.strip().split(' ')[0][3:] for line in f.readlines()]
    return classes

# Real-time scene classification generator for Flask streaming
def gen_scene_frames(interval=0.5):
    global camera
    with camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(0)
    model = load_places365()
    classes = get_places365_classes()
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    last_capture_time = 0
    scene_name = ""
    while True:
        with camera_lock:
            success, frame = camera.read()
        if not success:
            break
        current_time = time.time()
        if current_time - last_capture_time >= interval:
            # Convert frame (BGR) to PIL Image (RGB)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            input_img = transform(pil_img).unsqueeze(0)
            with torch.no_grad():
                output = model(input_img)
                scene_index = torch.argmax(output)
                scene_name = classes[scene_index]
            last_capture_time = current_time
        # Display scene name on frame
        cv2.putText(frame, f"Scene: {scene_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        # Yield frame in multipart/x-mixed-replace format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/scene_feed')
def scene_feed():
    return Response(gen_scene_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/scene')
def scene_index():
    # Simple HTML page to show the scene classification video stream
    return '''
    <html>
      <head>
        <title>Real-time Scene Classification</title>
      </head>
      <body>
        <h1>Real-time Scene Classification</h1>
        <img src="/scene_feed" width="800" />
      </body>
    </html>
    '''

def classify_scene(img_path=None):
    """
    Classify scene from an image file.
    Returns the predicted scene name.
    """
    model = load_places365()
    classes = get_places365_classes()
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    input_img = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_img)
        scene_index = torch.argmax(output)
        scene_name = classes[scene_index]
    return scene_name

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
