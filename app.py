import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ---------------- Page Config ----------------
st.set_page_config(page_title="Cat vs Dog", page_icon="🐾", layout="centered")

# ---------------- CSS Fix ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #F7F6F2;
}

/* Hide Streamlit menu */
#MainMenu, footer, header {visibility: hidden;}

/* Titles */
.main-title {
    font-size: 2.1rem;
    font-weight: 500;
    color: #1a1a1a;
    text-align: center;
    margin-bottom: 0.2rem;
}

.sub-title {
    font-size: 0.9rem;
    color: #888;
    text-align: center;
    margin-bottom: 2rem;
}

/* Upload container */
[data-testid="stFileUploader"] {
    background: white;
    border-radius: 16px;
    padding: 1.2rem;
    box-shadow: 0 2px 20px rgba(0,0,0,0.06);
}

/* Remove dark drag header */
[data-testid="stFileUploader"] section {
    background: transparent !important;
    border: none !important;
}

/* Remove drag text */
[data-testid="stFileUploader"] section div:first-child {
    display: none !important;
}

/* File name color fix */
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] div,
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p {
    color: #000000 !important;
}

/* Result box */
.result-box {
    background: white;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
    box-shadow: 0 2px 20px rgba(0,0,0,0.06);
}

.result-label {
    font-size: 2.4rem;
    font-weight: 500;
    color: #1a1a1a;
}

.result-conf {
    font-size: 0.9rem;
    color: #999;
    margin-top: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Load Model ----------------
@st.cache_resource
def load_cat_dog_model():
    return load_model("cat_dog_model.keras")

model = load_cat_dog_model()

# ---------------- Header ----------------
st.markdown('<div class="main-title">🐾 Cat vs Dog Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload an image to identify</div>', unsafe_allow_html=True)

# ---------------- File Upload ----------------
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png", "jfif", "webp"],
    label_visibility="collapsed"
)

# ---------------- Prediction ----------------
if uploaded_file is not None:

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(uploaded_file, width=380)

    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((64, 64))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.reshape(img_array, (1, 64, 64, 3))

    result = model.predict(img_array, verbose=0)
    score = result[0][0]

    if score >= 0.5:
        label = "Dog 🐶"
        confidence = score
    else:
        label = "Cat 🐱"
        confidence = 1 - score

    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">{label}</div>
        <div class="result-conf">{confidence*100:.2f}% confidence</div>
    </div>
    """, unsafe_allow_html=True)
