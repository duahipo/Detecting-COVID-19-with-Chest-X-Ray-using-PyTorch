import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import io
import os

# ======================================================
# UI – Title & Warning
# ======================================================
st.title("COVID-19 Detection from Chest X-Ray")
st.subheader("Deep Learning with PyTorch – Educational Use Only")
st.warning(
    "⚠️ This application is for educational purposes only "
    "and must NOT be used for medical diagnosis."
)

# ======================================================
# Device
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Running on: **{device}**")

# ======================================================
# Classes (MUST match training order)
# ======================================================
classes = ["Normal", "Viral Pneumonia", "COVID-19"]

# ======================================================
# Image preprocessing (MUST match training)
# ======================================================
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ======================================================
# Model path (ABSOLUTE PATH)
# ======================================================
MODEL_PATH = r"C:/Jupyter/Detecting COVID-19 with Chest X-Ray using PyTorch/model.pth"

# ======================================================
# Safe model loader
# ======================================================
@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found:\n{model_path}")
        return None

    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))

    try:
        checkpoint = torch.load(model_path, map_location=device)

        # Case 1: checkpoint contains state_dict
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])

        # Case 2: checkpoint is directly state_dict
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)

        else:
            raise ValueError("Unsupported checkpoint format")

        model.to(device)
        model.eval()

        # Sanity check
        with torch.no_grad():
            test_input = torch.zeros((1, 3, 224, 224)).to(device)
            test_output = model(test_input)
            assert test_output.shape[1] == len(classes)

        st.success("✅ Trained model loaded successfully")
        return model

    except Exception as e:
        st.error("❌ Failed to load the trained model")
        st.error(str(e))
        return None

# ======================================================
# Load model
# ======================================================
model = load_model(MODEL_PATH)

# ❗ Stop app if model is not loaded
if model is None:
    st.stop()

# ======================================================
# Image uploader
# ======================================================
uploaded_file = st.file_uploader(
    "Upload a Chest X-Ray image (PNG / JPG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    try:
        image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    except Exception:
        st.error("❌ Unable to read the uploaded file as an image.")
        st.stop()

    st.image(image, caption="Uploaded image", use_column_width=True)

    if st.button("Predict"):
        input_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)

        st.markdown("## 🔍 Prediction Result")
        st.write(f"**Class:** {classes[prediction.item()]}")
        st.write(f"**Confidence:** {confidence.item() * 100:.2f}%")

else:
    st.info("📤 Upload an image to enable prediction.")
