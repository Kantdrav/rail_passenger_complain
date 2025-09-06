import io
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from fastapi import FastAPI, File, UploadFile
from PIL import Image

# -----------------------------
# Step 1: Initialize FastAPI
# -----------------------------
app = FastAPI()

# -----------------------------
# Step 2: Define transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# -----------------------------
# -----------------------------
# Step 3: Load model & classes
# -----------------------------
checkpoint_path = os.path.join(os.path.dirname(__file__), "model.pth")
checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
class_names = checkpoint["classes"]

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))
model.load_state_dict(checkpoint["model_state"])
model.eval()


# -----------------------------
# Step 4: Prediction function
# -----------------------------
def predict_image(image: Image.Image):
    img = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]

# -----------------------------
# Step 5: API Endpoints
# -----------------------------
@app.get("/")
def home():
    return {"message": "🚆 Rail Complaint Classification API is running"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    class_name = predict_image(image)
    return {"prediction": class_name}
