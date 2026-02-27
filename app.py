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
        font-size: 2rem; font-weight: 500; color: #1a1a1a;
        text-align: center; letter-spacing: -0.5px; margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 0.9rem; color: #999; text-align: center;
        font-weight: 300; margin-bottom: 2rem;
    }
    .result-box {
        background: white; border-radius: 16px; padding: 2rem;
        text-align: center; margin-top: 1.5rem;
        box-shadow: 0 2px 20px rgba(0,0,0,0.06);
    }
    .result-label { font-size: 2.4rem; font-weight: 500; color: #1a1a1a; letter-spacing: -1px; }
    .result-conf { font-size: 0.88rem; color: #bbb; margin-top: 0.3rem; }

    /* Pura file uploader widget hide karo */
    [data-testid="stFileUploader"] { display: none !important; }

    /* Custom upload button */
    .upload-btn {
        display: block;
        width: 100%;
        background: #1a1a1a;
        color: white !important;
        text-align: center;
        padding: 1.2rem;
        border-radius: 14px;
        font-size: 1rem;
        font-weight: 400;
        letter-spacing: 0.2px;
        cursor: pointer;
        margin-bottom: 1.5rem;
    }

    /* Streamlit button style override */
    [data-testid="stFileUploader"] { display: none !important; }

    div[data-testid="stButton"] button {
        background-color: #1a1a1a !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.8rem 2rem !important;
        font-size: 1rem !important;
        width: 100% !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    div[data-testid="stButton"] button:hover {
        background-color: #333 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_cat_dog_model():
    return load_model('cat_dog_model.keras')

model = load_cat_dog_model()

st.markdown('<p class="main-title">🐾 Cat vs Dog</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload an image to identify</p>', unsafe_allow_html=True)

# Hidden uploader
uploaded_file = st.file_uploader(
    "x",
    type=["jpg", "jpeg", "png", "jfif", "webp"],
    label_visibility="collapsed",
    key="uploader"
)

# Custom visible button — triggers the hidden uploader via JS
st.markdown("""
<label for="uploader" style="
    display: block;
    background: #1a1a1a;
    color: white;
    text-align: center;
    padding: 1.1rem;
    border-radius: 14px;
    font-size: 1rem;
    cursor: pointer;
    margin-bottom: 1rem;
">
    ☁️ &nbsp; Choose Image
</label>
<script>
    const label = document.querySelector('label[for="uploader"]');
    if (label) {
        label.addEventListener('click', () => {
            const input = document.querySelector('input[type="file"]');
            if (input) input.click();
        });
    }
</script>
""", unsafe_allow_html=True)

if uploaded_file is not None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(uploaded_file, width=380)

    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((64, 64))
    img_array = image.img_to_array(img) / 255.0
    img_array = img_array.reshape(1, 64, 64, 3)

    result = model.predict(img_array, verbose=0)
    score = result[0][0]

    if score >= 0.5:
        label      = "Dog 🐶"
        confidence = score
    else:
        label      = "Cat 🐱"
        confidence = 1 - score

    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">{label}</div>
        <div class="result-conf">{confidence*100:.1f}% confidence</div>
    </div>
    """, unsafe_allow_html=True)
