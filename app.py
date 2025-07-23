import tensorflow as tf
import streamlit as st
import numpy as np
import cv2

# ==========================
# Load TFLite Model
# ==========================
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ==========================
# Class Labels
# ==========================
class_names = ['Mask', 'No Mask']

# ==========================
# Streamlit UI
# ==========================
st.title('😷 Face Mask Detection')

uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Read and display the image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(image, caption='Uploaded Image', channels='RGB', use_column_width=True)

    # Preprocess image
    input_shape = input_details[0]['shape'][1:3]  # e.g., (224, 224)
    image_resized = cv2.resize(image, tuple(input_shape))
    image_normalized = image_resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(image_normalized, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_index = np.argmax(output_data)
    confidence = output_data[0][predicted_index]

    # Show result
    if predicted_index == 0:
        st.success(f"✅ The person **IS wearing a mask**")
    else:
        st.error(f"❌ The person **is NOT wearing a mask**")
