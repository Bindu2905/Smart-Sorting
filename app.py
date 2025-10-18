from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import base64
import io
from PIL import Image

app = Flask(__name__)

# ✅ Load model
MODEL_PATH = "model/fruit_model.h5"
model = load_model(MODEL_PATH)
CLASS_NAMES = ["Fresh", "Rotten"]

def is_blurry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < 120

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        image_data = data["image"].split(",")[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("RGB")
        image = np.array(image)
        image = cv2.resize(image, (224, 224))

        if is_blurry(image):
            return jsonify({"result": "Blurry Image — Please recapture"})

        img = image / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        label = CLASS_NAMES[int(np.round(prediction[0][0]))]

        return jsonify({"result": label})
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"result": "Error in prediction"})

if __name__ == "__main__":
    app.run(debug=True)