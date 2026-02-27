import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Page config
st.set_page_config(
    page_title="PawScan — Cat vs Dog Detector",
    page_icon="🐾",
    layout="centered"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Syne:wght@300;400;500&display=swap');

    *, html, body {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
    }

    .stApp {
        background-color: #0d0d0d;
        background-image:
            radial-gradient(ellipse 80% 60% at 50% -10%, rgba(255,180,60,0.08) 0%, transparent 70%),
            radial-gradient(ellipse 60% 40% at 80% 100%, rgba(255,100,80,0.06) 0%, transparent 60%);
        min-height: 100vh;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── HERO HEADER ── */
    .hero {
        text-align: center;
        padding: 3.5rem 1rem 2rem;
        position: relative;
    }

    .hero-eyebrow {
        font-family: 'Syne', sans-serif;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.25em;
        text-transform: uppercase;
        color: #f0a030;
        margin-bottom: 1rem;
        opacity: 0.9;
    }

    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: clamp(3rem, 8vw, 5.5rem);
        font-weight: 900;
        color: #ffffff;
        line-height: 0.95;
        letter-spacing: -2px;
        margin-bottom: 1rem;
    }

    .hero-title span {
        color: #f0a030;
    }

    .hero-subtitle {
        font-family: 'Syne', sans-serif;
        font-size: 0.88rem;
        font-weight: 300;
        color: #666;
        letter-spacing: 0.05em;
    }

    .hero-divider {
        width: 40px;
        height: 2px;
        background: #f0a030;
        margin: 1.5rem auto;
        border-radius: 2px;
    }

    /* ── UPLOAD ZONE ── */
    .upload-label {
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #444;
        text-align: center;
        margin-bottom: 0.6rem;
    }

    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.03);
        border: 1.5px dashed rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 2rem 1.5rem;
        transition: border-color 0.3s;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: rgba(240,160,48,0.4);
    }

    [data-testid="stFileUploader"] label {
        display: none;
    }

    /* ── IMAGE PREVIEW ── */
    .img-frame {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 20px;
        padding: 1.2rem;
        margin-top: 1.5rem;
        display: flex;
        justify-content: center;
    }

    [data-testid="stImage"] img {
        border-radius: 14px !important;
        display: block;
        max-width: 100%;
    }

    /* ── RESULT CARD ── */
    .result-wrapper {
        margin-top: 2rem;
        padding: 0 0.5rem;
    }

    .result-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #141414 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        padding: 2.5rem 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .result-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #f0a030, #ff6040);
        border-radius: 24px 24px 0 0;
    }

    .result-tag {
        display: inline-block;
        font-size: 0.65rem;
        font-weight: 500;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: #f0a030;
        background: rgba(240,160,48,0.1);
        border: 1px solid rgba(240,160,48,0.2);
        padding: 0.3rem 0.9rem;
        border-radius: 100px;
        margin-bottom: 1.2rem;
    }

    .result-animal {
        font-family: 'Playfair Display', serif;
        font-size: clamp(2.8rem, 10vw, 4.5rem);
        font-weight: 900;
        color: #ffffff;
        letter-spacing: -2px;
        line-height: 1;
        margin-bottom: 0.4rem;
    }

    .result-emoji {
        font-size: 2.5rem;
        display: block;
        margin-bottom: 0.8rem;
    }

    .result-conf-label {
        font-size: 0.7rem;
        font-weight: 400;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #555;
        margin-bottom: 0.6rem;
    }

    .confidence-bar-bg {
        background: rgba(255,255,255,0.06);
        border-radius: 100px;
        height: 6px;
        width: 100%;
        max-width: 280px;
        margin: 0 auto 0.7rem;
        overflow: hidden;
    }

    .confidence-bar-fill {
        height: 100%;
        border-radius: 100px;
        background: linear-gradient(90deg, #f0a030, #ff6040);
        transition: width 0.6s ease;
    }

    .result-conf-value {
        font-family: 'Playfair Display', serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: #f0a030;
        letter-spacing: -0.5px;
    }

    /* ── FOOTER ── */
    .footer-note {
        text-align: center;
        font-size: 0.7rem;
        color: #333;
        letter-spacing: 0.1em;
        padding: 2.5rem 0 1.5rem;
    }

</style>
""", unsafe_allow_html=True)


# ── Load Model ──
@st.cache_resource
def load_cat_dog_model():
    return load_model('cat_dog_model.keras')

model = load_cat_dog_model()


# ── Hero Header ──
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">AI-Powered Vision</div>
    <div class="hero-title">Paw<span>Scan</span></div>
    <div class="hero-divider"></div>
    <div class="hero-subtitle">Drop an image — we'll tell you what's inside</div>
</div>
""", unsafe_allow_html=True)


# ── Upload ──
st.markdown('<div class="upload-label">Upload your image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png", "jfif", "webp"],
    label_visibility="collapsed"
)


# ── Predict ──
if uploaded_file is not None:

    # Image preview
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown('<div class="img-frame">', unsafe_allow_html=True)
        st.image(uploaded_file, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Preprocess
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((64, 64))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 64, 64, 3)

    # Predict
    result = model.predict(img_array, verbose=0)
    score = result[0][0]

    if score >= 0.5:
        animal = "Dog"
        emoji = "🐶"
        confidence = score
    else:
        animal = "Cat"
        emoji = "🐱"
        confidence = 1 - score

    conf_pct = confidence * 100
    bar_width = f"{conf_pct:.1f}%"

    # Result card
    st.markdown(f"""
    <div class="result-wrapper">
        <div class="result-card">
            <div class="result-tag">Detection Complete</div>
            <span class="result-emoji">{emoji}</span>
            <div class="result-animal">{animal}</div>
            <div class="result-conf-label">Confidence Score</div>
            <div class="confidence-bar-bg">
                <div class="confidence-bar-fill" style="width: {bar_width};"></div>
            </div>
            <div class="result-conf-value">{conf_pct:.1f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ──
st.markdown('<div class="footer-note">Powered by CNN · Built with Streamlit</div>', unsafe_allow_html=True)
