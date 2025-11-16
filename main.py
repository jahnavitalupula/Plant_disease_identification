import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# -----------------------------
# Load TFLite Model
# -----------------------------
# Load your TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -----------------------------
# Prediction Function
# -----------------------------
def model_prediction(test_image):
    image = Image.open(test_image).resize((128, 128))
    img_array = np.array(image, dtype=np.float32)

    # Adjust shape for batch size: (1,128,128,3)
    img_array = np.expand_dims(img_array, axis=0)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    prediction = interpreter.get_tensor(output_details[0]['index'])

    return np.argmax(prediction)

# -----------------------------
# Streamlit UI
# -----------------------------

st.sidebar.title("AgriSens")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Display banner image
banner = Image.open("Diseases.png")
st.image(banner)

# HOME PAGE
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>SMART DISEASE DETECTION</h1>", unsafe_allow_html=True)

# DISEASE RECOGNITION PAGE
elif app_mode == "DISEASE RECOGNITION":
    st.header("DISEASE RECOGNITION")

    test_image = st.file_uploader("Choose an Image:")

    if st.button("Show Image") and test_image:
        st.image(test_image, use_column_width=True)

    if st.button("Predict") and test_image:
        st.snow()
        st.write("Our Prediction")

        result_index = model_prediction(test_image)

        # CLASS LABELS
        class_name = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
            'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]

        st.success(f"Model is predicting: **{class_name[result_index]}**")
