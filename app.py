from flask import Flask, request, render_template, send_file, abort
from detect_objects import detect_objects
from classify_scene import classify_scene
import os
from werkzeug.utils import secure_filename
import shutil

app = Flask(__name__, template_folder='.')  # index.html is in the same folder as this script
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

if __name__ == "__main__":
    app.run(debug=True)
