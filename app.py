import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.set_page_config(page_title="Cat vs Dog", page_icon="🐾", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background-color: #F7F6F2; }

    .main-title {
        font-size: 2.2rem; font-weight: 500; color: #1a1a1a;
        text-align: center; letter-spacing: -0.5px; margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 0.95rem; color: #888; text-align: center;
        font-weight: 300; margin-bottom: 2.5rem;
    }
    .result-box {
        background: white; border-radius: 16px; padding: 2rem;
        text-align: center; margin-top: 1.5rem;
        box-shadow: 0 2px 20px rgba(0,0,0,0.06);
    }
    .result-label { font-size: 2.5rem; font-weight: 500; color: #1a1a1a; letter-spacing: -1px; }
    .result-conf { font-size: 0.9rem; color: #aaa; margin-top: 0.3rem; font-weight: 300; }

    .file-name-box {
        background: white;
        border-radius: 0 0 12px 12px;
        padding: 0.6rem 1rem;
        font-size: 0.85rem;
        color: #1a1a1a !important;
        margin-top: -1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Upload outer card */
    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 16px;
        padding: 0;
        box-shadow: 0 2px 20px rgba(0,0,0,0.06);
        overflow: hidden;
    }
    [data-testid="stFileUploader"] label { display: none; }

    /* Hide default file name row - we show our own */
    [data-testid="stFileUploader"] section > div > div:not([data-testid="stFileUploaderDropzone"]) {
        display: none !important;
    }

    [data-testid="stImage"] img { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_cat_dog_model():
    return load_model('cat_dog_model.keras')

model = load_cat_dog_model()

st.markdown('<p class="main-title">🐾 Cat vs Dog</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload an image to identify</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "jfif", "webp"], label_visibility="collapsed")

# Apna file name dikhao - dark text guaranteed
if uploaded_file is not None:
    st.markdown(f'<div class="file-name-box">📄 {uploaded_file.name}</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(uploaded_file, width=400)

    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((64, 64))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 64, 64, 3)

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
        <div class="result-conf">{confidence*100:.1f}% confidence</div>
    </div>
    """, unsafe_allow_html=True)
