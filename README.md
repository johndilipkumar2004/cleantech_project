- [Demo Link](https://drive.google.com/file/d/1mdykXxwXzWL_1nRKlKjuHpRcew7UgnF0/view?usp=drivesdk)
- [Code](https://github.com/johndilipkumar2004/cleantech_project.git)
# cleantech_projet
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions, MobileNetV2

st.set_page_config(page_title="CleanTech Waste Classifier", layout="centered")
st.title("♻️ CleanTech: Waste Material Classifier")

MODEL_PATH = "mobilenet_v2"
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Mapping from ImageNet classes to waste categories
WASTE_MAP = {
    "banana": "🍌 Organic Waste",
    "apple": "🍏 Organic Waste",
    "orange": "🍊 Organic Waste",
    "plastic_bag": "♻️ Recyclable - Plastic",
    "plastic_bottle": "♻️ Recyclable - Plastic",
    "water_bottle": "♻️ Recyclable - Plastic",
    "milk_can": "🧴 Recyclable - Metal",
    "can": "♻️ Recyclable - Metal",
    "packet": "🗑️ General Waste",
    "cardboard": "📦 Recyclable - Paper",
    "paper_towel": "🧻 Compostable Paper",
    "newspaper": "📰 Recyclable - Paper",
    "carton": "📦 Recyclable - Paper",
    "trash_can": "🗑️ General Waste"
}


def predict_image(img):
    img = img.resize((224, 224))
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    decoded = decode_predictions(preds, top=1)[0][0]
    label = decoded[1].lower()
    category = WASTE_MAP.get(label, "🗑️ General Waste")
    print("Predicted Label:", label)  # Debug print
    return label, category


option = st.radio("Choose Input Method:", ("Upload Image", "Use Camera"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload a waste image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        label, category = predict_image(img)
        st.markdown(f"**🔍 Detected object:** `{label}`")
        st.markdown(f"**📦 Waste Category:** {category}")

elif option == "Use Camera":
    st.info("Click 'Start Camera' and then 'Capture Photo' to classify waste")
    start = st.button("Start Camera")

    if start:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        capture = st.button("📸 Capture Photo")

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access camera")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", caption="Live Camera Feed")

            if capture:
                img = Image.fromarray(frame_rgb)
                label, category = predict_image(img)
                stframe.image(frame_rgb, caption=f"Detected: {label}\nCategory: {category}", channels="RGB")
                break

        cap.release()
        cv2.destroyAllWindows()
