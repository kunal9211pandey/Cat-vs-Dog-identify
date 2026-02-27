import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.set_page_config(page_title="Cat vs Dog", page_icon="🐾", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&display=swap');

    * { font-family: 'DM Sans', sans-serif; }
    .stApp { background-color: #F7F6F2; }

    #MainMenu, footer, header { visibility: hidden; }

    .main-title {
        font-size: 2rem;
        font-weight: 500;
        color: #1a1a1a;
        text-align: center;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 0.9rem;
        color: #999;
        text-align: center;
        font-weight: 300;
        margin-bottom: 2rem;
    }
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
        letter-spacing: -1px;
    }
    .result-conf {
        font-size: 0.88rem;
        color: #bbb;
        margin-top: 0.3rem;
    }
    .filename-row {
        font-size: 0.85rem;
        color: #555;
        padding: 0.5rem 0 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_cat_dog_model():
    return load_model('cat_dog_model.keras')

model = load_cat_dog_model()

# Header
st.markdown('<p class="main-title">🐾 Cat vs Dog</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload an image to identify</p>', unsafe_allow_html=True)

# Upload - plain, no CSS tricks
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png", "jfif", "webp"],
    label_visibility="collapsed"
)

if uploaded_file is not None:

    # Show uploaded file name cleanly
    st.markdown(f'<p class="filename-row">📄 {uploaded_file.name}</p>', unsafe_allow_html=True)

    # Show image centered
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(uploaded_file, width=380)

    # Prepare and predict
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((64, 64))
    img_array = image.img_to_array(img) / 255.0
    img_array = img_array.reshape(1, 64, 64, 3)

    result = model.predict(img_array, verbose=0)
    score = result[0][0]

    if score >= 0.5:
        label    = "Dog 🐶"
        confidence = score
    else:
        label    = "Cat 🐱"
        confidence = 1 - score

    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">{label}</div>
        <div class="result-conf">{confidence*100:.1f}% confidence</div>
    </div>
    """, unsafe_allow_html=True)
