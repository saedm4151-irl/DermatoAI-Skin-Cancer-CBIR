# --------------------------------------------------------------------------
# Project: Dermato-AI: Skin Lesion Classification & CBIR System
# Author: Saed (https://github.com/saedm4151-irl)
# Description: Clinical decision-support tool using ResNet-50 & CBIR.
# --------------------------------------------------------------------------

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import zipfile
import io
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="SkinScan AI | Research Dashboard", layout="wide")

# --- CONFIGURATION ---
HF_MODEL_REPO = "saedm4151-irl/dermato-ai-resnet50"
DATASET_ZIP = "ISIC2018_Complete_Dataset.zip" # Must be in the same folder as app.py

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['Melanoma (MEL)', 'Melanocytic Nevi (NV)', 'Basal Cell Carcinoma (BCC)', 
               'Actinic Keratosis (AKIEC)', 'Benign Keratosis (BKL)', 
               'Dermatofibroma (DF)', 'Vascular Lesion (VASC)']

@st.cache_resource
def load_all_resources():
    # 1. Download Model and CBIR files from Hugging Face Model Repo
    model_weight_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="Updated_best.pth")
    features_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="features.npy")
    filenames_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="filenames.npy")

    # 2. Load Model Architecture
    model = models.resnet50()
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 7)
    )
    
    # 3. Load Weights
    state_dict = torch.load(model_weight_path, map_location=DEVICE)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.to(DEVICE).eval()
    
    # 4. Load CBIR Database
    features_db = np.load(features_path)
    filenames_db = np.load(filenames_path, allow_pickle=True)
    
    return model, features_db, filenames_db

# Initialize Resources
model, features_db, filenames_db = load_all_resources()
backbone = torch.nn.Sequential(*list(model.children())[:-1])

@st.cache_resource
def get_zip_ref():
    if not os.path.exists(DATASET_ZIP):
        st.error(f"Dataset archive {DATASET_ZIP} not found! Please ensure it is uploaded to the Space.")
        return None
    return zipfile.ZipFile(DATASET_ZIP, 'r')

zf = get_zip_ref()

# --- PREDICTION LOGIC ---
def get_prediction(img):
    base_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    tta_transforms = [lambda x: x, transforms.functional.hflip, transforms.functional.vflip]
    all_probs = []
    
    with torch.no_grad():
        for t in tta_transforms:
            aug_img = t(img)
            tensor = base_transform(aug_img).unsqueeze(0).to(DEVICE)
            output = model(tensor)
            all_probs.append(torch.softmax(output, dim=1))
            
    avg_probs = torch.stack(all_probs).mean(0)
    conf, idx = torch.max(avg_probs, 1)
    
    original_tensor = base_transform(img).unsqueeze(0).to(DEVICE)
    query_vec = backbone(original_tensor).view(1, -1).detach().cpu().numpy()
    
    return idx.item(), conf.item(), avg_probs[0].cpu().numpy(), query_vec

# --- UI LAYOUT ---
st.title("SkinScan AI: Clinical Reference Tool")
st.markdown("AI-driven diagnostic prediction & Content-Based Image Retrieval (CBIR).")
st.divider()

uploaded_file = st.sidebar.file_uploader("Upload a Dermoscopic Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    query_img = Image.open(uploaded_file).convert("RGB")
    
    with st.spinner('Analyzing patterns...'):
        pred_idx, confidence, probs, query_vec = get_prediction(query_img)

    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.image(query_img, caption="Uploaded Image", use_container_width=True)
        
    with col2:
        st.header(f"Prediction: **{CLASS_NAMES[pred_idx]}**")
        st.subheader(f"Confidence: {confidence*100:.1f}%")
        st.progress(confidence)
        prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
        st.bar_chart(prob_dict)

    st.divider()
    st.subheader("Similar Historical Cases (CBIR Results)")
    
    # Calculate Similarity
    similarities = cosine_similarity(query_vec, features_db).flatten()
    top_k_indices = similarities.argsort()[-3:][::-1]
    
    match_cols = st.columns(3)
    if zf:
        for i, idx in enumerate(top_k_indices):
            with match_cols[i]:
                filename = filenames_db[idx]
                internal_path = f"ISIC2018_Task3_Training_Input/{filename}.jpg"
                
                try:
                    img_data = zf.read(internal_path)
                    match_img = Image.open(io.BytesIO(img_data))
                    st.image(match_img, use_container_width=True)
                    st.write(f"**Similarity Score:** {similarities[idx]:.4f}")
                    st.caption(f"Reference ID: {filename}")
                except Exception:
                    st.error(f"Error loading {filename}")
else:
    st.info("Please upload an image in the sidebar to begin analysis.")
