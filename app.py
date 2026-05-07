import streamlit as st
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from PIL import Image


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Potato Leaf Disease Detection",
    layout="wide"
)


# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.main {
    background-color: #f5f5f5;
}

.title-box {
    background: linear-gradient(to right, #1b5e20, #66bb6a);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    color: white;
    margin-bottom: 20px;
}

.upload-box {
    background-color: #e6dcc8;
    padding: 10px;
    border-radius: 10px;
}

.prediction-box {
    background-color: #2e7d32;
    padding: 15px;
    border-radius: 10px;
    color: white;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
}

.conf-box {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)


# =========================
# LOAD MODEL
# =========================

model = load_model("cnn_model.h5")


# =========================
# CLASS NAMES
# IMPORTANT:
# CHECK ORDER USING
# print(train_generator.class_indices)
# =========================

class_names = [
    'Early Blight',
    'Late Blight',
    'Healthy'
]


# =========================
# TITLE
# =========================

st.markdown("""
<div class="title-box">
    <h1>🥔 🌿 Potato Leaf Disease Detection</h1>
    <p>Upload a potato leaf image to detect plant health</p>
</div>
""", unsafe_allow_html=True)


# =========================
# FILE UPLOADER
# =========================

st.markdown(
    '<div class="upload-box">📤 Upload Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "",
    type=['jpg', 'jpeg', 'png']
)


# =========================
# PREDICTION
# =========================

if uploaded_file is not None:

    # Open Image
    img = Image.open(uploaded_file)

    # Display Image
    col1, col2 = st.columns([1.2,1])

    with col1:
        st.image(
            img,
            caption='🖼️ Uploaded Image',
            use_container_width=True
        )

    # Resize Image
    img = img.resize((224,224))

    # Convert to Array
    img_array = image.img_to_array(img)

    # Normalize
    img_array = img_array / 255.0

    # Expand Dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    # Predicted Index
    predicted_index = np.argmax(prediction)

    # Predicted Class
    predicted_class = class_names[predicted_index]

    # Confidence
    confidence = np.max(prediction) * 100


    # =========================
    # RIGHT SIDE PANEL
    # =========================

    with col2:

        st.markdown("""
        <div class="conf-box">
        <h2>📊 Confidence Level</h2>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(confidence))

        st.write(f"Confidence: {confidence:.2f}%")

        if confidence > 90:
            st.success("🔥 High Confidence")

        elif confidence > 70:
            st.warning("⚡ Medium Confidence")

        else:
            st.error("❌ Low Confidence")


    # =========================
    # PREDICTION OUTPUT
    # =========================

    st.markdown(
        f'''
        <div class="prediction-box">
        ⚠️ Prediction: {predicted_class} 🍃
        </div>
        ''',
        unsafe_allow_html=True
    )