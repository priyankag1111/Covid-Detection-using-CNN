import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

# ==========================================
# LOAD TFLITE MODEL
# ==========================================

interpreter = tf.lite.Interpreter(
    model_path="covid_xray_model.tflite"
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class Labels
classes = ['COVID', 'Normal', 'Viral Pneumonia']

# ==========================================
# STREAMLIT UI
# ==========================================

st.title("COVID-19 Detection from Chest X-rays")

st.write("Upload a chest X-ray image to predict whether it is COVID, Normal, or Viral Pneumonia.")

# ==========================================
# FILE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=['jpg', 'jpeg', 'png']
)

# ==========================================
# PREDICTION
# ==========================================

if uploaded_file is not None:

    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display uploaded image
    st.image(
        img,
        caption='Uploaded Chest X-ray',
        use_column_width=True
    )

    # Resize image
    resized_img = cv2.resize(img, (128, 128))

    # Normalize image
    resized_img = resized_img / 255.0

    # Expand dimensions for model input
    input_data = np.expand_dims(
        resized_img,
        axis=0
    ).astype(np.float32)

    # ==========================================
    # TFLITE MODEL INFERENCE
    # ==========================================

    interpreter.set_tensor(
        input_details[0]['index'],
        input_data
    )

    interpreter.invoke()

    prediction = interpreter.get_tensor(
        output_details[0]['index']
    )

    # Get prediction
    predicted_index = np.argmax(prediction)

    predicted_class = classes[predicted_index]

    confidence = float(
        np.max(prediction) * 100
    )

    # ==========================================
    # DISPLAY RESULTS
    # ==========================================

    st.success(
        f"Prediction: {predicted_class}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )
