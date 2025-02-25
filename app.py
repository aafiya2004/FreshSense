from flask import Flask, request, jsonify
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the trained PyTorch model
MODEL_PATH = "model.pth"  # Update with your actual model path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model class (Update this according to your architecture)
class YourModelClass(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define your model layers here
    def forward(self, x):
        return torch.randn(1, 2)  # Mock output (Replace with actual forward pass)

# Load model
model = YourModelClass()  # Replace with actual model
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Define freshness labels
fresh_labels = ['Fresh', 'Spoiled']

# Image preprocessing function
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match model input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(img_path).convert("RGB")  # Load image
    return transform(image).unsqueeze(0).to(device)  # Add batch dimension

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
        fresh_pred = model(img_tensor)

    # Convert predictions to probabilities
    fresh_probs = F.softmax(fresh_pred, dim=1).cpu().numpy()[0]

    # Get predicted freshness and confidence score
    pred_fresh_idx = torch.argmax(fresh_pred, axis=1).item()
    pred_freshness = fresh_labels[pred_fresh_idx]
    fresh_confidence = fresh_probs[pred_fresh_idx] * 100

    # Decision logic
    if pred_fresh_idx == 0 and fresh_confidence > 85:  # Fresh with high confidence
        decision = "✅ Supply to Market"
    elif pred_fresh_idx == 1 and fresh_confidence > 85:  # Spoiled with high confidence
        decision = "❌ Discard"
    else:
        decision = "⚠ Further Inspection Needed"

    # Return JSON response
    return jsonify({
        "Predicted Quality": pred_freshness,
        "Confidence Score": f"{fresh_confidence:.2f}%",
        "Decision": decision
    })

if __name__ == "__main__":
    app.run(debug=True)
