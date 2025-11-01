import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = load_model("src/defect_model.keras")

st.title("🧬 Defect Detection in Manufacturing Lines")
st.write("Upload an image to check if it’s **Defective** or **Normal**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((150, 150))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    result = "🧫 Defective Sample" if prediction[0][0] < 0.5 else "✅ Normal Sample"

    st.subheader(result)
