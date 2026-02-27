import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ---------------- Page Config ----------------
st.set_page_config(page_title="Cat vs Dog", page_icon="🐾", layout="centered")

# ---------------- Professional CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #F4F6F8;
}

/* Hide Streamlit default elements */
#MainMenu, footer, header {visibility: hidden;}

/* Main Card */
.main-card {
    background: white;
    padding: 2.2rem;
    border-radius: 18px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.05);
    max-width: 520px;
    margin: auto;
    margin-top: 2rem;
}

/* Title */
.title {
    text-align: center;
    font-size: 1.9rem;
    font-weight: 600;
    color: #111;
    margin-bottom: 0.3rem;
}

.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 0.95rem;
    margin-bottom: 1.8rem;
}

/* Uploader Styling */
[data-testid="stFileUploader"] {
    border: 2px dashed #e5e7eb;
    border-radius: 12px;
    padding: 1rem;
    background: #fafafa;
}

/* Remove big dark drag header */
[data-testid="stFileUploader"] section {
    background: transparent !important;
    border: none !important;
}

/* File name color */
[data-testid="stFileUploader"] * {
    color: #111 !important;
}

/* Image */
.preview-img img {
    border-radius: 12px;
}

/* Result */
.result {
    margin-top: 1.5rem;
    padding: 1.2rem;
    background: #f9fafb;
    border-radius: 12px;
    text-align: center;
}

.result-label {
    font-size: 1.8rem;
    font-weight: 600;
    color: #111;
}

.result-conf {
    font-size: 0.9rem;
    color: #6b7280;
    margin-top: 0.2rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Load Model ----------------
@st.cache_resource
def load_cat_dog_model():
    return load_model("cat_dog_model.keras")

model = load_cat_dog_model()

# ---------------- Card Layout ----------------
st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.markdown('<div class="title">🐾 Cat vs Dog Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image to predict</div>', unsafe_allow_html=True)

# Upload
uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png", "jfif", "webp"],
    label_visibility="collapsed"
)

# ---------------- Prediction ----------------
if uploaded_file is not None:

    st.markdown('<div class="preview-img">', unsafe_allow_html=True)
    st.image(uploaded_file, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Preprocess
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((64, 64))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.reshape(img_array, (1, 64, 64, 3))

    # Predict
    result = model.predict(img_array, verbose=0)
    score = result[0][0]

    if score >= 0.5:
        label = "Dog 🐶"
        confidence = score
    else:
        label = "Cat 🐱"
        confidence = 1 - score

    st.markdown(f"""
    <div class="result">
        <div class="result-label">{label}</div>
        <div class="result-conf">{confidence*100:.2f}% confidence</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
