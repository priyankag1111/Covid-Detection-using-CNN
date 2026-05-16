import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

# Load Model
model = tf.keras.models.load_model('covid_xray_model.tflite')

classes = ['COVID', 'Normal', 'Viral Pneumonia']

# Title
st.title("COVID-19 Detection from Chest X-rays")

# Upload Image
uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(file_bytes, 1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img, caption='Uploaded X-ray')

    img = cv2.resize(img, (128,128))

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    predicted_class = classes[np.argmax(prediction)]

    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {predicted_class}")

    st.info(f"Confidence: {confidence:.2f}%")

