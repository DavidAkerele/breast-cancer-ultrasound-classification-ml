import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import streamlit as st

import config
from dataset import apply_clahe
from models import get_model

# Page config
st.set_page_config(
    page_title="OncoVision AI | Dissertation Medical Diagnostic UI",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark clinical aesthetics
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #000000, #0a0a0a);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 2rem;
    }
    .metric-card {
        background: rgba(10, 10, 10, 0.8);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    .disclaimer {
        font-size: 0.85rem;
        color: #94a3b8;
        background: rgba(5, 5, 5, 0.9);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #38bdf8;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if os.path.exists(config.CHECKPOINT_PATH):
        checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
        model_name = checkpoint.get("model_name", "custom_cnn")
        model = get_model(model_name=model_name, num_classes=config.NUM_CLASSES, pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = get_model(model_name="custom_cnn", num_classes=config.NUM_CLASSES, pretrained=False)
    
    model = model.to(config.DEVICE)
    model.eval()
    return model

model = load_model()

val_transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Sidebar Dissertation Context
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/ribbon--v1.png", width=64)
    st.header("Dissertation Research")
    st.markdown("---")
    st.markdown("**Topic:** Automated Breast Cancer Ultrasound Classification using Deep Convolutional Neural Networks")
    st.markdown("**Backend:** PyTorch & OpenCV CLAHE")
    st.markdown("**Framework:** Streamlit / FastAPI")
    st.markdown("---")
    use_clahe = st.toggle("Apply CLAHE Contrast Enhancement", value=True, help="Equalizes lesion contrast")
    st.markdown("<div class='disclaimer'>⚠️ <b>Academic Research Prototype:</b> Not intended for primary clinical diagnosis without radiologist verification.</div>", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 style="color: #38bdf8; margin: 0;">🎗️ OncoVision AI Diagnostic Suite</h1>
    <p style="color: #cbd5e1; font-size: 1.1rem; margin-top: 8px;">Dissertation Prototype: Automated Breast Ultrasound Classification & Tissue Analysis</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("1. Scan Input & Preprocessing")
    uploaded_file = st.file_uploader("Upload Breast Ultrasound Study (PNG, JPEG, TIFF)", type=["png", "jpg", "jpeg", "tif"])
    
    if not uploaded_file:
        st.info("💡 Upload a breast ultrasound image or select a benchmark sample below.")
        if st.button("Load Benchmark Malignant Sample"):
            sample_path = os.path.join(config.DATA_DIR, "val", "malignant", "sample_0.png")
            if os.path.exists(sample_path):
                uploaded_file = open(sample_path, "rb")

with col2:
    st.subheader("2. Deep Learning Diagnostic Analysis")
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            uploaded_file.seek(0)
            img_pil = Image.open(uploaded_file).convert("RGB")
            img_np = np.array(img_pil)
        else:
            img_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        enhanced_np = apply_clahe(img_np) if use_clahe else img_np

        # Display Comparison
        scol1, scol2 = st.columns(2)
        with scol1:
            st.image(img_np, caption="Original Scan", use_column_width=True)
        with scol2:
            st.image(enhanced_np, caption="CLAHE Enhanced View", use_column_width=True)

        # PyTorch Inference
        img_pil = Image.fromarray(enhanced_np)
        tensor = val_transform(img_pil).unsqueeze(0).to(config.DEVICE)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1).squeeze(0)
            pred_idx = torch.argmax(probs).item()
            pred_class = config.CLASS_NAMES[pred_idx]
            confidence = probs[pred_idx].item()

        st.markdown("---")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            color = "#10b981" if pred_class == "benign" else "#f43f5e"
            st.markdown(f"<div class='metric-card'><h4 style='color: #94a3b8; margin:0;'>AI Verdict</h4><h2 style='color: {color}; margin: 8px 0;'>{pred_class.upper()}</h2></div>", unsafe_allow_html=True)
        with res_col2:
            st.markdown(f"<div class='metric-card'><h4 style='color: #94a3b8; margin:0;'>Model Confidence</h4><h2 style='color: #38bdf8; margin: 8px 0;'>{confidence*100:.2f}%</h2></div>", unsafe_allow_html=True)

        st.markdown("### Class Activation Probability")
        b_prob = float(probs[0].item())
        m_prob = float(probs[1].item())
        st.write(f"**Benign:** {b_prob*100:.2f}%")
        st.progress(b_prob)
        st.write(f"**Malignant:** {m_prob*100:.2f}%")
        st.progress(m_prob)
    else:
        st.warning("Awaiting breast ultrasound upload to execute neural classification.")
