import os
import torch
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from PIL import Image

# -----------------------------
# Step 1: Transform images
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# -----------------------------
# Step 2: Load dataset
# -----------------------------
dataset = datasets.ImageFolder("dataset", transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# -----------------------------
# Step 3: Use pretrained model
# -----------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # Updated per torchvision warning
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(dataset.classes))  # Automatically handle number of classes

# -----------------------------
# Step 4: Loss and optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# Step 5: Train model
# -----------------------------
for epoch in range(5):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} completed")

# -----------------------------
# Step 6: Predict new image
# -----------------------------
# Use the correct path to your image
image = "/home/kantdravi/Desktop/rail/rail_madad/images/image.png"
if not os.path.exists(image):
    raise FileNotFoundError(f"Image not found: {image}")

img = Image.open(image).convert('RGB')
img = transform(img).unsqueeze(0)  # Add batch dimension

model.eval()
with torch.no_grad():
    pred = model(img)
    class_idx = torch.argmax(pred, dim=1).item()
    class_name = dataset.classes[class_idx]
    print("Forward to:", class_name)
