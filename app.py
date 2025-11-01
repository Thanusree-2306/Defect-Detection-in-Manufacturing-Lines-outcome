import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2

st.set_page_config(page_title="Defect Detection", layout="centered")

st.title("🧠 Defect Detection in Manufacturing Lines")
st.write("Upload an image to check if it’s **Defective** or **Normal.**")

# Load model
model = load_model("defect_model.keras")

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image to array and preprocess
    img = np.array(image.resize((224, 224))) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)
    result = "Defective" if prediction[0][0] > 0.5 else "Normal"

    st.subheader(f"✅ Prediction: {result}")
