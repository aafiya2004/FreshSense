from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import json

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the trained model
MODEL_PATH = "model.keras"  # Update with your actual model path
model = tf.keras.models.load_model(MODEL_PATH)

# Load class labels from saved JSON file
with open("class_indices.json", "r") as f:
    class_labels = json.load(f)
class_labels_list = list(class_labels.keys())  # Convert to list for indexing

# Image preprocessing function (same as in Colab)
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(160, 160))  # Resize
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route("/", methods=["GET"])
def home():
    return "Flask app is running! Use /predict to classify images."

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Securely save the uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Preprocess the image
    img_array = preprocess_image(file_path)

    # Get model prediction
    prediction = model.predict(img_array)

    # Get class index and probability
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_prob = prediction[0][predicted_class_index]  # Confidence score
    predicted_class = class_labels_list[predicted_class_index]

    # Determine freshness or rottenness
    threshold = 10  # Threshold for classification
    freshness_percentage = 0.0
    rotten_percentage = 0.0
    decision = "Discard"

    if "Fresh" in predicted_class:
        predicted_quality = "Fresh"
        freshness_percentage = predicted_class_prob * 100
        decision = "Supply to market" if freshness_percentage >= threshold else "Further inspection needed"
    elif "Rotten" in predicted_class:
        predicted_quality = "Rotten"
        rotten_percentage = (1 - predicted_class_prob) * 100
        decision = "Discard" if rotten_percentage >= threshold else "Further inspection needed"
    else:
        predicted_quality = "Unknown Class"
        decision = "Further inspection needed"

    # Return JSON response
    return jsonify({
        "Predicted Quality": predicted_quality,
        "Confidence Score": f"{freshness_percentage if predicted_quality == 'Fresh' else rotten_percentage:.2f}%",
        "Decision": decision
    })

if __name__ == "__main__":
    app.run(debug=True)
