import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

st.title("Handwritten Digit Recognizer")

model = tf.keras.models.load_model("model.h5")

uploaded_file = st.file_uploader("Upload an image of a digit", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28)

    st.image(image, caption='Uploaded Image', width=200)
    
    prediction = model.predict(img_array)
    st.write(f"Predicted Digit: {np.argmax(prediction)}")
