import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "model/animal_model_2.keras"
IMG_SIZE = (128, 128)

CLASS_NAMES = [
    "butterfly",
    "cat",
    "chicken",
    "cow",
    "dog",
    "elephant",
    "horse",
    "sheep",
    "squirrel"
]

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Page Config
st.set_page_config(page_title="Animal Classification", layout="centered")

st.title("Animal Image Classification")
st.write("Upload an image and the model will predict the animal.")

# Image uploader
uploaded_file = st.file_uploader("Choose an img file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    image = image.resize(IMG_SIZE)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    predictions = model.predict(image_array)
    predicted_index = np.argmax(predictions)
    confidence = predictions[0][predicted_index]

    predicted_label = CLASS_NAMES[predicted_index]

    # Show Result
    st.subheader("Prediction Result")
    st.write(f"Predicted animal: **{predicted_label}**")
    st.write(f"Confidence: **{confidence:.2f}**")
