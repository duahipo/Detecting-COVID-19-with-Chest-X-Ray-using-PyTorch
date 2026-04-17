import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import io

# Title / subtitle
st.title("COVID-19 Detection from Chest X-Ray")
st.subheader("Deep Learning with PyTorch – Educational Use Only")
st.warning("This application is for educational purposes only and must not be used for medical diagnosis.")

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Classes
classes = ["Normal", "Viral Pneumonia", "COVID-19"]

# Preprocessing (must match training)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Cached model loader
@st.cache(allow_output_mutation=True)
def load_model(path="model.pth"):
    # build model architecture
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    try:
        state = torch.load(path, map_location=device)
        # Accept both state_dict and full checkpoint
        if isinstance(state, dict) and 'state_dict' in state:
            model.load_state_dict(state['state_dict'])
        else:
            model.load_state_dict(state)
        st.info(f"Model loaded from '{path}'")
    except FileNotFoundError:
        st.error(f"Model file '{path}' not found. Place the trained model as '{path}' in the app folder.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
    model = model.to(device)
    model.eval()
    return model

model = load_model("model.pth")

# Uploader
uploaded_file = st.file_uploader("Upload a Chest X-Ray image (PNG / JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(io.BytesIO(uploaded_file.read())).convert('RGB')
    except Exception as e:
        st.error("Unable to read the uploaded file as an image.")
        st.stop()

    st.image(image, caption="Uploaded image", use_column_width=True)

    if st.button("Predict"):
        # Preprocess
        input_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            predicted_class = classes[pred.item()]
            confidence = conf.item()

        st.markdown("**Prediction**")
        st.write(f"Class: {predicted_class}")
        st.write(f"Confidence: {confidence*100:.2f}%")
else:
    st.info("Upload an image to enable prediction.")
