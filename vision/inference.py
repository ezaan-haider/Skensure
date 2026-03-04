import torch
import torch.nn as nn
from torchvision.models import convnext_tiny
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os

# -------------------
# CONFIG
# -------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "convnext_tiny.pth")
NUM_CLASSES = 5

CLASS_NAMES = [
    'Vi Shingles',
    'Vi Chickenpox',
    'Healthy',
    'Acne And Rosacea',
    'Atopic Dermatitis'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------
# LOAD MODEL
# -------------------
model = convnext_tiny(weights=None)
in_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(in_features, NUM_CLASSES)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# -------------------
# TRANSFORM
# -------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------
# TRAINING PREPROCESSING
# -------------------
def preprocess_image(image):
    img_array = np.array(image)

    img_yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_array = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    img_array = cv2.GaussianBlur(img_array, (3, 3), 0)

    return Image.fromarray(img_array)

# -------------------
# PREDICTION FUNCTION
# -------------------
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")

    image = preprocess_image(image)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)

        temp = 3.0
        probs = torch.softmax(outputs/temp, dim=1)

        predicted_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_idx].item()

    return CLASS_NAMES[predicted_idx], confidence

# -------------------
# TEST IMAGE
# -------------------
