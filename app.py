import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

st.title("Handwritten Digit Recognizer")

model = tf.keras.models.load_model("model.h5")

uploaded_file = st.file_uploader("Upload an image of a digit (28x28 grayscale)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = ImageOps.invert(image)  # Invert the image if needed
    image = image.resize((28, 28))  # Resize to 28x28
    img_array = np.array(image) / 255.0  # Normalize
    img_array = img_array.reshape(1, 28, 28)

    st.image(image, caption='Uploaded Image', width=200)
    
    prediction = model.predict(img_array)
    st.write(f"Predicted Digit: {np.argmax(prediction)}")