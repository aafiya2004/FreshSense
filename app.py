from flask import Flask, request, jsonify
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from werkzeug.utils import secure_filename
from model import load_model  # Import model loading function

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model
MODEL_PATH = "model.pth"
model, device = load_model(MODEL_PATH)

# Define image preprocessing function (Ensure it matches training pipeline)
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Match ResNet input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # ResNet standard
    ])
    image = Image.open(img_path).convert("RGB")  # Convert to RGB
    return transform(image).unsqueeze(0).to(device)

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

    # Securely save uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Preprocess the image
    img_tensor = preprocess_image(file_path)

    # Get model prediction
    with torch.no_grad():
        freshness_output = model(img_tensor)

    # Convert to probabilities
    probs = F.softmax(freshness_output, dim=1).cpu().numpy()[0]

    # Get predicted freshness label
    pred_fresh_idx = torch.argmax(freshness_output, axis=1).item()
    fresh_labels = ["Fresh", "Spoiled"]
    pred_freshness = fresh_labels[pred_fresh_idx]
    confidence = probs[pred_fresh_idx] * 100

    # Decision logic
    if confidence > 50:
        decision = f"{confidence:.2f}% confident - {'✅ Supply to Market' if pred_fresh_idx == 0 else '❌ Discard'}"
        remaining_confidence = 100 - confidence
        if remaining_confidence > 0:
            decision += f", {remaining_confidence:.2f}% ⚠ Further Inspection Needed"
    else:
        decision = "⚠ Further Inspection Needed"


    # Return JSON response
    return jsonify({
        "Predicted Quality": pred_freshness,
        "Confidence Score": f"{confidence:.2f}%",
        "Decision": decision
    })

if __name__ == "__main__":
    app.run(debug=True)
