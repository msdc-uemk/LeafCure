import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

# ====================
# Setup
# ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((300, 300)),   # match training script
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ====================
# Load Models (state_dict version)
# ====================
def load_model(path, num_classes):
    model = models.efficientnet_b3(weights=None)  # same arch as training
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Load each model with correct output size
crop_model = load_model("models/crop_classifier.pth", num_classes=3)
tomato_model = load_model("models/tomatodisease_model.pth", num_classes=10)
potato_model = load_model("models/potatodisease_model.pth", num_classes=3)
bellpepper_model = load_model("models/belldisease_model.pth", num_classes=2)

# ====================
# Class Labels
# ====================
# Order must match ImageFolder alphabetical order during training
crop_classes = ["bellpepper", "potato", "tomato"]

tomato_classes = [
    "bacterialspot", "earlyblight", "healthy", "lateblight", "leafmold",
    "mosaicvirus", "septorailleafspot", "spidermites", "targetspot", 
    "yellowleafcurlvirus"
]

potato_classes = ["earlyblight", "lateblight", "healthy"]
bellpepper_classes = ["bacterialspot", "healthy"]

# ====================
# Prediction Helper
# ====================
def predict(image, model, classes):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)  # ✅ probabilities
        confidence, predicted = torch.max(probs, 1)
    return classes[predicted.item()], confidence.item()

# ====================
# Routes
# ====================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})

    # Save uploaded file
    filepath = os.path.join("static/uploads", file.filename)
    os.makedirs("static/uploads", exist_ok=True)
    file.save(filepath)

    # Open image
    image = Image.open(filepath).convert("RGB")

    # Step 1: Identify crop
    crop, _ = predict(image, crop_model, crop_classes)

    # Step 2: Route to disease model
    if crop == "tomato":
        disease, conf = predict(image, tomato_model, tomato_classes)
    elif crop == "potato":
        disease, conf = predict(image, potato_model, potato_classes)
    elif crop == "bellpepper":
        disease, conf = predict(image, bellpepper_model, bellpepper_classes)
    else:
        disease, conf = "Unknown", 0.0

    return jsonify({
        "crop": crop,   
        "disease": disease,
        "confidence": conf,   # ✅ send confidence to JS
        "file_path": filepath
    })

if __name__ == "__main__":
    app.run(debug=True)
