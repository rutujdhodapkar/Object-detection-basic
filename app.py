from flask import Flask, request, render_template, send_file, abort, Response
from detect_objects import detect_objects
from classify_scene import classify_scene
import os
from werkzeug.utils import secure_filename
import shutil
import cv2
import threading
import time
from ultralytics import YOLO

app = Flask(__name__, template_folder='.')  # index.html is in the same folder as this script
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for camera thread
camera = None
camera_lock = threading.Lock()

def gen_frames():
    global camera
    with camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(0)
    while True:
        with camera_lock:
            success, frame = camera.read()
        if not success:
            break
        else:
            # Save frame to a temp file for detection/classification
            temp_path = os.path.join(UPLOAD_FOLDER, "temp_cam_frame.jpg")
            cv2.imwrite(temp_path, frame)

            # Run detection and classification every 0.5 seconds
            try:
                objects, detected_img_path = detect_objects(temp_path)
                scene = classify_scene(temp_path)
                # Read the detected image (with boxes) for streaming
                detected_frame = cv2.imread(detected_img_path)
                # Overlay scene and objects as text
                label = f"Scene: {scene} | Objects: {', '.join(objects)}"
                cv2.putText(detected_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                ret, buffer = cv2.imencode('.jpg', detected_frame)
            except Exception as e:
                # If detection/classification fails, just show the raw frame
                ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.5)  # Capture and predict every 0.5 seconds

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    scene = None
    objects = None
    detected_img = None
    output_img_url = None

    if request.method == "POST":
        if "image" not in request.files:
            result = "No file part"
        else:
            file = request.files["image"]
            if file.filename == "":
                result = "No selected file"
            else:
                filename = secure_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)

                # Run detection and classification
                objects, detected_img_path = detect_objects(filepath)
                scene = classify_scene(filepath)
                detected_img = detected_img_path

                # Move detected image to uploads folder with a unique name
                detected_img_filename = "output_detected_" + filename
                detected_img_upload_path = os.path.join(UPLOAD_FOLDER, detected_img_filename)
                try:
                    shutil.move(detected_img, detected_img_upload_path)
                except Exception as e:
                    # If move fails, try copy as fallback
                    try:
                        shutil.copy(detected_img, detected_img_upload_path)
                        os.remove(detected_img)
                    except Exception as e2:
                        result = f"Error saving detected image: {e2}"
                        detected_img_upload_path = None

                if detected_img_upload_path and os.path.exists(detected_img_upload_path):
                    output_img_url = "/output_image/" + detected_img_filename
                else:
                    output_img_url = None

    return render_template("index.html", scene=scene, objects=objects, output_img_url=output_img_url, result=result)

@app.route("/output_image/<filename>")
def output_image(filename):
    # Serve from uploads folder, not cwd
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        abort(404, description="Image not found")
    return send_file(file_path, mimetype="image/jpeg")

# --- Real-time YOLOv8 camera stream (from realtime.py algorithm) ---
# This endpoint and generator provide real-time object detection using YOLOv8

# Global camera object and lock for thread safety (for YOLOv8 stream)
realtime_camera = None
realtime_camera_lock = threading.Lock()

def gen_frames_realtime():
    global realtime_camera
    with realtime_camera_lock:
        if realtime_camera is None:
            realtime_camera = cv2.VideoCapture(0)
    model = YOLO('yolov8s.pt')
    while True:
        with realtime_camera_lock:
            success, frame = realtime_camera.read()
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
    return Response(gen_frames_realtime(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live_camera')
def live_camera():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Note: The HTML for the camera button and video stream should be placed in index.html.
# Example (add to index.html where appropriate):
# <button id="openCameraBtn" onclick="document.getElementById('cam').style.display='block';this.style.display='none';">Open Camera</button>
# <div id="cam" style="display:none;">
#     <img src="/video_feed" width="800" />
# </div>
# You can use JavaScript to show/hide the camera stream on button click.

if __name__ == "__main__":
    app.run(debug=True, threaded=True)