import os
import torch
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torch.nn.functional as F

# -----------------------------
# Step 0: Cleanup corrupted images
# -----------------------------
dataset_root = "dataset"

for root, _, files in os.walk(dataset_root):
    for file in files:
        fpath = os.path.join(root, file)
        try:
            img = Image.open(fpath)
            img.verify()  # Check if it's a valid image
        except Exception as e:
            print(f"❌ Bad file removed: {fpath} ({e})")
            os.remove(fpath)

# -----------------------------
# Step 1: Define transforms (with augmentation)
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# Step 2: Safe dataset loader
# -----------------------------
class SafeImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        good_imgs = []
        for path, label in self.samples:
            try:
                img = Image.open(path)
                img.verify()
                good_imgs.append((path, label))
            except Exception as e:
                print(f"❌ Removing bad file: {path} ({e})")
        self.samples = good_imgs
        self.imgs = self.samples

    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception as e:
            print(f"⚠️ Skipping bad file at index {index}: {self.imgs[index][0]} ({e})")
            return torch.zeros(3, 224, 224), 0

dataset = SafeImageFolder(dataset_root, transform=test_transform)  # base dataset for class info

# -----------------------------
# Step 3: Split dataset
# -----------------------------
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Apply transforms separately
train_dataset.dataset.transform = train_transform
test_dataset.dataset.transform = test_transform

# -----------------------------
# Step 3.1: Balanced Sampler
# -----------------------------
class_counts = [len([s for s in dataset.samples if s[1] == i]) for i in range(len(dataset.classes))]
print("Class counts:", dict(zip(dataset.classes, class_counts)))

# Compute weights (inverse frequency)
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = [class_weights[label] for _, label in train_dataset]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# -----------------------------
# Step 4: Load pretrained model
# -----------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(dataset.classes))

# -----------------------------
# Step 5: Loss and optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()  # sampler already balances classes
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# -----------------------------
# Step 6: Evaluate function
# -----------------------------
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# -----------------------------
# Step 7: Train model
# -----------------------------
for epoch in range(250):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()

    train_acc = evaluate(model, train_loader)
    test_acc = evaluate(model, test_loader)
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, "
          f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

# -----------------------------
# Step 8: Predict new image
# -----------------------------
image_path = "/home/kantdravi/Desktop/rail_passenger_complain/ai_model/images/image.png"

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")

img = Image.open(image_path).convert('RGB')
img = test_transform(img).unsqueeze(0)

model.eval()
with torch.no_grad():
    pred = model(img)
    probs = F.softmax(pred, dim=1)

    # Top-3 predictions
    top_p, top_class = torch.topk(probs, 3, dim=1)

    print("\nTop Predictions:")
    for i in range(3):
        cls = dataset.classes[top_class[0][i].item()]
        conf = top_p[0][i].item() * 100
        print(f"  {cls}: {conf:.2f}%")

    # All class probabilities
    print("\nAll probabilities:")
    for cls, p in zip(dataset.classes, probs[0]):
        print(f"  {cls}: {p.item()*100:.2f}%")

# -----------------------------
# Step 9: Save model + classes
# -----------------------------
save_path = os.path.join(os.path.dirname(__file__), "model.pth")
torch.save({
    "model_state": model.state_dict(),
    "classes": dataset.classes
}, save_path)

print(f"\n✅ Model trained and saved as {save_path}")
