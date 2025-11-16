import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# -----------------------------
# Load TFLite Model
# -----------------------------

MODEL_PATH = "model.tflite"

if not os.path.exists(MODEL_PATH):
    st.error("❌ model.tflite not found in repository!")
else:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -----------------------------
# Preprocessing + Prediction
# -----------------------------
def model_prediction(test_image):

    image = Image.open(test_image).resize((128, 128))
    image = np.array(image, dtype=np.float32)

    # If model expects normalization
    if image.max() > 1:
        image = image / 255.0

    input_data = np.expand_dims(image, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    result_index = np.argmax(output)

    return result_index


# -----------------------------
# UI
# -----------------------------
st.sidebar.title("SmartGrow AI")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Display banner
if os.path.exists("Diseases.png"):
    st.image("Diseases.png", use_column_width=True)

# -----------------------------
# HOME PAGE
# -----------------------------
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>SMART DISEASE DETECTION</h1>",
                unsafe_allow_html=True)

# -----------------------------
# PREDICTION PAGE
# -----------------------------
elif app_mode == "DISEASE RECOGNITION":
    st.header("DISEASE RECOGNITION")

    test_image = st.file_uploader("Choose an Image:")

    if test_image and st.button("Show Image"):
        st.image(test_image, use_column_width=True)

    if test_image and st.button("Predict"):
        st.snow()
        st.write("Our Prediction")

        result_index = model_prediction(test_image)

        # Labels
        class_name = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
            'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
        ]

        st.success(f"Model predicts: **{class_name[result_index]}** 🌿")

